import torch, dgl
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch.glob import AvgPooling, SumPooling

class MultiHeadAttentionLayer(nn.Module):
    """
    멀티헤드 어텐션 레이어를 구현한 클래스
    각 노드의 특징을 학습하고 그래프의 노드 및 엣지 간의 관계를 모델링
    """

    def __init__(self, num_input_feats, num_output_feats, num_heads):
        """
        MultiHeadAttentionLayer의 초기화 메서드

        Args:
            num_input_feats (int): 입력 특징의 수
            num_output_feats (int): 출력 특징의 수
            num_heads (int): 어텐션 헤드의 수
        """
        super(MultiHeadAttentionLayer, self).__init__()

        self.num_output_feats = num_output_feats
        self.num_heads = num_heads

        # 각 어텐션 헤드를 위한 쿼리, 키, 값 및 엣지 가중치 행렬 정의
        self.Q = nn.Linear(num_input_feats, self.num_output_feats * self.num_heads)
        self.K = nn.Linear(num_input_feats, self.num_output_feats * self.num_heads)
        self.V = nn.Linear(num_input_feats, self.num_output_feats * self.num_heads)
        self.E = nn.Linear(num_input_feats, self.num_output_feats * self.num_heads)

    def propagate_attention(self, g):
        """
        그래프의 모든 엣지에 대해 어텐션 점수를 계산하고 업데이트하는 함수

        Args:
            g (DGLGraph): 입력 그래프
        """
        # 어텐션 점수를 계산하기 위해 노드 간의 내적을 수행
        g.apply_edges(lambda edges: {"score":  edges.src['K_h'] * edges.dst['Q_h']})  # 내적 연산
        g.apply_edges(lambda edges: {"score": (edges.data["score"] / np.sqrt(self.num_output_feats)).clamp(-5.0, 5.0)})  # 스케일 조정 및 클리핑
        g.apply_edges(lambda edges: {"score":  edges.data['score'] * edges.data['proj_e']})  # 엣지 가중치 적용
        g.apply_edges(lambda edges: {"e_out":  edges.data["score"]})  # 결과 저장
        g.apply_edges(lambda edges: {"score":  torch.exp((edges.data["score"].sum(-1, keepdim=True)).clamp(-5.0, 5.0))})  # softmax 계산 및 클리핑
        # 노드 간의 어텐션 결과를 집계하여 업데이트
        g.update_all(fn.u_mul_e('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))  # dot product 및 합계 연산
        g.update_all(fn.copy_e('score', 'score'), fn.sum('score', 'z'))

    def forward(self, g, node_feats, edge_feats):
        """
        멀티헤드 어텐션 레이어의 순전파 연산을 정의

        Args:
            g (DGLGraph): 입력 그래프
            node_feats (torch.Tensor): 노드 특징
            edge_feats (torch.Tensor): 엣지 특징

        Returns:
            torch.Tensor, torch.Tensor: 노드 및 엣지의 업데이트된 특징
        """
        with g.local_scope():  # 로컬 범위 내에서 그래프 데이터 변경
            # 입력 특징을 멀티헤드 어텐션용으로 변환
            node_feats_q = self.Q(node_feats)
            node_feats_k = self.K(node_feats)
            node_feats_v = self.V(node_feats)
            edge_feats_e = self.E(edge_feats)

            # 노드와 엣지에 멀티헤드 어텐션 특징을 저장
            g.ndata['Q_h'] = node_feats_q.view(-1, self.num_heads, self.num_output_feats)
            g.ndata['K_h'] = node_feats_k.view(-1, self.num_heads, self.num_output_feats)
            g.ndata['V_h'] = node_feats_v.view(-1, self.num_heads, self.num_output_feats)
            g.edata['proj_e'] = edge_feats_e.view(-1, self.num_heads, self.num_output_feats)

            # 어텐션 전파 수행
            self.propagate_attention(g)

            # 업데이트된 노드 및 엣지 특징 계산
            h_out = g.ndata['wV'] / (g.ndata['z'] + torch.full_like(g.ndata['z'], 1e-8))  # 분모에 작은 값을 추가하여 안정성을 확보
            e_out = g.edata['e_out']

        return h_out, e_out

class GraphTransformerModule(nn.Module):
    """
    그래프 변환 모듈을 정의한 클래스
    각 그래프 레이어에서 노드와 엣지의 특징을 변환
    """
    def __init__(self, num_hidden_channels, residual=True, num_attention_heads=4, dropout_rate=0.1):
        """
        GraphTransformerModule의 초기화 메서드

        Args:
            num_hidden_channels (int): 숨겨진 특징 차원의 수
            residual (bool): 잔차 연결 사용 여부
            num_attention_heads (int): 어텐션 헤드의 수
            dropout_rate (float): 드롭아웃 비율
        """
        super(GraphTransformerModule, self).__init__()
        self.activ_fn = nn.SiLU()
        self.residual = residual
        self.num_attention_heads = num_attention_heads
        self.dropout_rate = dropout_rate

        self.num_hidden_channels, self.num_output_feats = num_hidden_channels, num_hidden_channels

        # 레이어 정규화 및 멀티헤드 어텐션 레이어 초기화
        self.layer_norm1_node_feats = nn.LayerNorm(self.num_output_feats)
        self.layer_norm1_edge_feats = nn.LayerNorm(self.num_output_feats)
        self.layer_norm2_node_feats = nn.LayerNorm(self.num_output_feats)
        self.layer_norm2_edge_feats = nn.LayerNorm(self.num_output_feats)

        self.mha_module = MultiHeadAttentionLayer(
            self.num_hidden_channels, 
            self.num_output_feats // self.num_attention_heads, 
            self.num_attention_heads
        )

        # 출력 레이어 초기화
        self.O_node_feats = nn.Linear(self.num_output_feats, self.num_output_feats)
        self.O_edge_feats = nn.Linear(self.num_output_feats, self.num_output_feats)

        # 드롭아웃을 포함한 MLP 구성
        dropout = nn.Dropout(p=self.dropout_rate) if self.dropout_rate > 0.0 else nn.Identity()
        self.node_feats_MLP = nn.ModuleList([
            nn.Linear(self.num_output_feats, self.num_output_feats * 2, bias=False),
            self.activ_fn,
            dropout,
            nn.Linear(self.num_output_feats * 2, self.num_output_feats, bias=False)
        ])

        self.edge_feats_MLP = nn.ModuleList([
            nn.Linear(self.num_output_feats, self.num_output_feats * 2, bias=False),
            self.activ_fn,
            dropout,
            nn.Linear(self.num_output_feats * 2, self.num_output_feats, bias=False)
        ])

    def forward(self, g, node_feats, edge_feats):
        """
        GraphTransformerModule의 순전파 연산을 정의

        Args:
            g (DGLGraph): 입력 그래프
            node_feats (torch.Tensor): 노드 특징
            edge_feats (torch.Tensor): 엣지 특징

        Returns:
            torch.Tensor, torch.Tensor: 업데이트된 노드 및 엣지 특징
        """
        # 첫 번째 잔차 연결을 위한 특징 저장
        node_feats_in1 = node_feats
        edge_feats_in1 = edge_feats

        # 첫 번째 레이어 정규화
        node_feats = self.layer_norm1_node_feats(node_feats)
        edge_feats = self.layer_norm1_edge_feats(edge_feats)

        # 멀티헤드 어텐션 적용
        node_attn_out, edge_attn_out = self.mha_module(g, node_feats, edge_feats)

        # 어텐션 출력 재구성
        node_feats = node_attn_out.view(-1, self.num_output_feats)
        edge_feats = edge_attn_out.view(-1, self.num_output_feats)

        # 드롭아웃 적용
        node_feats = F.dropout(node_feats, self.dropout_rate, training=self.training)
        edge_feats = F.dropout(edge_feats, self.dropout_rate, training=self.training)

        # 출력 특징 계산
        node_feats = self.O_node_feats(node_feats)
        edge_feats = self.O_edge_feats(edge_feats)

        if self.residual:
            # 첫 번째 잔차 연결
            node_feats = node_feats_in1 + node_feats
            edge_feats = edge_feats_in1 + edge_feats

        # 두 번째 잔차 연결을 위한 특징 저장
        node_feats_in2 = node_feats
        edge_feats_in2 = edge_feats

        # 두 번째 레이어 정규화
        node_feats = self.layer_norm2_node_feats(node_feats)
        edge_feats = self.layer_norm2_edge_feats(edge_feats)

        # MLP 적용
        for layer in self.node_feats_MLP:
            node_feats = layer(node_feats)

        for layer in self.edge_feats_MLP:
            edge_feats = layer(edge_feats)

        if self.residual:
            # 두 번째 잔차 연결
            node_feats = node_feats_in2 + node_feats
            edge_feats = edge_feats_in2 + edge_feats

        return node_feats, edge_feats

class PredictionScore(nn.Module):
    """
    그래프를 기반으로 예측 점수를 계산하는 모델
    """

    def __init__(self, in_size, emb_size, edge_size, num_layers, dropout_ratio=0.2):
        """
        PredictionScore의 초기화 메서드

        Args:
            in_size (int): 입력 노드 특징의 크기
            emb_size (int): 임베딩 크기
            edge_size (int): 엣지 특징의 크기
            num_layers (int): 그래프 변환 모듈의 레이어 수
            dropout_ratio (float): 드롭아웃 비율
        """
        super(PredictionScore, self).__init__()
        # 노드 및 엣지 특징 인코더 정의
        self.node_encoder = nn.Linear(in_size, emb_size)
        self.edge_encoder = nn.Linear(edge_size, emb_size)

        # 레이어 정규화
        self.node_norm = nn.LayerNorm(emb_size)
        self.edge_norm = nn.LayerNorm(emb_size)

        # Graph Transformer 모듈 초기화
        self.blocks = nn.ModuleList(
            [
                GraphTransformerModule(
                    num_hidden_channels=emb_size,
                    residual=True,
                    num_attention_heads=8,
                    dropout_rate=dropout_ratio
                )
                for _ in range(num_layers)
            ]
        )

        # 다양한 입력 특징에 대한 MLP 초기화
        self.mlp_morgan = nn.Sequential(
            nn.Linear(1024, emb_size),
            nn.BatchNorm1d(emb_size),
            nn.ELU(),
            nn.Dropout(p=dropout_ratio),
            nn.Linear(emb_size, emb_size),
        )
        self.mlp_avalon = nn.Sequential(
            nn.Linear(512, emb_size),
            nn.BatchNorm1d(emb_size),
            nn.ELU(),
            nn.Dropout(p=dropout_ratio),
            nn.Linear(emb_size, emb_size),
        )
        self.mlp_erg = nn.Sequential(
            nn.Linear(315, emb_size),
            nn.BatchNorm1d(emb_size),
            nn.ELU(),
            nn.Dropout(p=dropout_ratio),
            nn.Linear(emb_size, emb_size),
        )
        self.mlp_fps = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.BatchNorm1d(emb_size),
            nn.ELU(),
            nn.Dropout(p=dropout_ratio),
            nn.Linear(emb_size, emb_size),
        )

        # 예측 점수를 위한 MLP 초기화
        self.mlp_score = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.BatchNorm1d(emb_size),
            nn.ELU(),
            nn.Dropout(p=dropout_ratio),
            nn.Linear(emb_size, 1),
        )

        self.sum_pooling = SumPooling()

    def forward(self, g, morgan, avalon, erg):
        """
        PredictionScore의 순전파 연산을 정의

        Args:
            g (DGLGraph): 입력 그래프
            morgan (torch.Tensor): Morgan fingerprint 특징
            avalon (torch.Tensor): Avalon fingerprint 특징
            erg (torch.Tensor): Erg fingerprint 특징

        Returns:
            torch.Tensor: 예측 점수
        """
        # 노드와 엣지의 초기 특징을 인코딩하고 정규화
        h = self.node_encoder(g.ndata['feat'])
        e = self.edge_encoder(g.edata['feat'])

        h = self.node_norm(h)
        e = self.edge_norm(e)

        # 각 Graph Transformer 모듈을 통과
        for layer in self.blocks:
            h, e = layer(g, h, e)

        # 추가적인 특징들을 MLP에 통과
        morgan = self.mlp_morgan(morgan)
        avalon = self.mlp_avalon(avalon)
        erg = self.mlp_erg(erg)

        # Fingerprint 특징을 합치고 MLP에 통과
        fps = morgan + avalon + erg
        fps = self.mlp_fps(fps)

        # 그래프의 노드 특징을 합쳐줌 (Sum Pooling)
        h = self.sum_pooling(g, h)

        # 그래프 특징과 fingerprint 특징을 합침
        h = h + fps

        # 최종 예측 점수를 계산
        score = self.mlp_score(h)

        return score
