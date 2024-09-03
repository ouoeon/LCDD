import dgl, numpy
import torch as th

import torch.nn as nn
import torch.nn.functional as F

def run_train_epoch(model, data_loader, optimizer, device='cpu'):
    """
    모델의 한 에폭(epoch) 동안 훈련을 실행하는 함수

    Args:
        model (nn.Module): 학습할 모델
        data_loader (DataLoader): 훈련 데이터 로더
        optimizer (torch.optim.Optimizer): 모델의 파라미터를 업데이트하는 옵티마이저
        device (str): 모델과 데이터를 이동시킬 장치로 기본값은 'cpu'

    Returns:
        float: 에폭 동안의 평균 손실값
    """
    model.train()  # 모델을 훈련 모드로 설정

    total_loss = 0  # 총 손실 초기화
    for batch_idx, batch_data in enumerate(data_loader):
        # 배치 데이터를 장치로 이동
        g, morgan, avalon, erg, true_score = batch_data
        g, morgan, avalon, erg, true_score = (
            g.to(device),
            morgan.to(device),
            avalon.to(device),
            erg.to(device),
            true_score.to(device),
        )

        # 모델의 예측값 계산
        pred_score = model(g, morgan, avalon, erg)

        # 평균 제곱 오차(MSE) 손실 계산
        loss = F.mse_loss(pred_score, true_score)

        # 역전파를 통해 모델 파라미터 업데이트
        optimizer.zero_grad(set_to_none=True)  # 그래디언트 초기화
        loss.backward()  # 손실의 그래디언트 계산
        optimizer.step()  # 옵티마이저를 통해 모델 파라미터 업데이트

        # 배치의 손실을 총 손실에 추가
        total_loss += loss.item() * g.batch_size
        th.cuda.empty_cache()  # CUDA 메모리 캐시를 비움

    # 전체 데이터셋에 대한 평균 손실을 반환
    return total_loss / len(data_loader.dataset)


def run_eval_epoch(model, data_loader, train=True, device='cpu'):
    """
    모델의 한 에폭(epoch) 동안 평가를 실행하는 함수

    Args:
        model (nn.Module): 평가할 모델
        data_loader (DataLoader): 평가 데이터 로더
        train (bool): 훈련 중인지 여부를 나타내는 플래그로 기본값은 True
        device (str): 모델과 데이터를 이동시킬 장치로 기본값은 'cpu'

    Returns:
        tuple: 에폭 동안의 평균 손실값, 실제 점수, 예측 점수
    """
    model.eval()  # 모델을 평가 모드로 설정

    true = []  # 실제 점수를 저장할 리스트
    pred = []  # 예측 점수를 저장할 리스트
    total_loss = 0  # 총 손실 초기화

    with th.no_grad():  # 평가 모드에서는 그래디언트를 계산하지 않음
        for batch_idx, batch_data in enumerate(data_loader):
            # 배치 데이터를 장치로 이동
            g, morgan, avalon, erg, true_score = batch_data
            g, morgan, avalon, erg, true_score = (
                g.to(device),
                morgan.to(device),
                avalon.to(device),
                erg.to(device),
                true_score.to(device),
            )

            # 모델의 예측값 계산
            pred_score = model(g, morgan, avalon, erg)

            # 평균 제곱 오차(MSE) 손실 계산
            loss = F.mse_loss(pred_score, true_score)

            # 배치의 손실을 총 손실에 추가
            total_loss += loss.item() * g.batch_size
            true.extend(true_score)  # 실제 점수를 리스트에 추가
            pred.extend(pred_score)  # 예측 점수를 리스트에 추가

            th.cuda.empty_cache()  # CUDA 메모리 캐시를 비움

    divisor = len(data_loader.dataset)  # 데이터셋의 전체 크기
    pred = th.tensor(pred)  # 예측 점수를 텐서로 변환
    true = th.tensor(true)  # 실제 점수를 텐서로 변환

    # 전체 데이터셋에 대한 평균 손실, 실제 점수 텐서, 예측 점수 텐서를 반환
    return total_loss / divisor, true, pred
