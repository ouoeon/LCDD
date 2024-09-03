import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import random
from sklearn.model_selection import train_test_split
from rdkit import Chem
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from rdkit import RDLogger
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

def set_seed(seed):
    ''' 
    랜덤 시드를 설정하여 결과의 재현성을 보장
    Args:
        seed (int): 설정할 시드 값
    '''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f'random seed is set {seed}.')

set_seed(42)

fname = "chemvae"
NUM_EPOCHS = 200
BATCH_SIZE = 512
LATENT_DIM = 290
LR = 0.001
DYN_LR = True
SMILES_COL_NAME = 'smiles'

df = pd.read_csv('./zinc_canon.csv')

train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

def tokenizer(smiles_string):
    '''
    SMILES 문자열을 토큰화
    Args:
        smiles_string (str): SMILES 문자열
    Returns:
        list: 문자 단위로 분리된 리스트
    '''
    return list(smiles_string)

def build_vocab(data):
    '''
    데이터에서 vocab과 inv_dict을 생성
    Args:
        data (DataFrame): SMILES 문자열이 포함된 데이터프레임
    Returns:
        tuple: vocab과 inv_dict
    '''
    vocab_ = set()
    smiles = list(data[SMILES_COL_NAME])
    for ex in smiles:
        for letter in tokenizer(ex):
            vocab_.add(letter)
    
    vocab = {'<PAD>': 0 }
    for i, letter in enumerate(vocab_):
        vocab[letter] = i + 1
    inv_dict = {num: char for char, num in vocab.items()}
    inv_dict[0] = ''
    return vocab, inv_dict

def make_one_hot(data, vocab, max_len=120):
    '''
    데이터를 원-핫 인코딩 형식으로 변환
    Args:
        data (list): SMILES 문자열 리스트
        vocab (dict): 어휘 사전
        max_len (int): 최대 문자열 길이
    Returns:
        np.ndarray: 원-핫 인코딩된 데이터
    '''
    data_one_hot = np.zeros((len(data), max_len, len(vocab)))
    for i, smiles in enumerate(data):
        smiles = tokenizer(smiles)
        smiles = smiles[:120] + ['<PAD>'] * (max_len - len(smiles))
        for j, char in enumerate(smiles):
            if char in vocab.keys():
                data_one_hot[i, j, vocab[char]] = 1
    return data_one_hot

def onehot_to_smiles(onehot, inv_vocab):
    '''
    원-핫 인코딩된 데이터를 SMILES 문자열로 변환
    Args:
        onehot (np.ndarray): 원-핫 인코딩된 데이터
        inv_vocab (dict): 인버스 사전
    Returns:
        str: SMILES 문자열
    '''
    return "".join(inv_vocab[let.item()] for let in onehot.argmax(axis=2)[0])

device = 'cuda' if torch.cuda.is_available() else 'cpu'

vocab, inv_dict = build_vocab(train_df)
vocab_size = len(vocab)
print(vocab)
print(inv_dict)

train_data = make_one_hot(train_df['smiles'], vocab)
print("Input train Data Shape", train_data.shape)

valid_data = make_one_hot(val_df['smiles'], vocab)
print("Input valid Data Shape", valid_data.shape)

train_properties = train_df[['logP', 'qed', 'SAS']].values
valid_properties = val_df[['logP', 'qed', 'SAS']].values

class MolecularDataset(torch.utils.data.Dataset):
    '''
    분자 데이터셋을 정의하는 클래스
    '''
    def __init__(self, data, properties):
        self.data = data
        self.properties = properties

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.properties[idx]

train_dataset = MolecularDataset(train_data, train_properties)
valid_dataset = MolecularDataset(valid_data, valid_properties)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=120, out_channels=9, kernel_size=9)
        self.conv2 = nn.Conv1d(in_channels=9, out_channels=9, kernel_size=9)
        self.conv3 = nn.Conv1d(in_channels=9, out_channels=10, kernel_size=11)
        self.encode_norm1 = nn.BatchNorm1d(9)
        self.encode_norm2 = nn.BatchNorm1d(9)
        self.encode_norm3 = nn.BatchNorm1d(10)
        self.flatten = nn.Flatten()
        self.encode_fc1 = nn.Linear(90, 196)
        self.encode_norm4 = nn.BatchNorm1d(196)
        self.z_mean_layer = nn.Linear(196, 196)
        self.z_log_var_layer = nn.Linear(196, 196)

    def forward(self, x):
        '''
        인코더의 순방향 전파
        Args:
            x (Tensor): 입력 데이터
        Returns:
            tuple: 잠재 변수의 평균(z_mean)과 로그 분산(z_log_var)
        '''
        x = self.encode_norm1(torch.tanh(self.conv1(x)))
        x = self.encode_norm2(torch.tanh(self.conv2(x)))
        x = self.encode_norm3(torch.tanh(self.conv3(x)))
        x = self.flatten(x)
        x = self.encode_norm4(self.encode_fc1(x))
        z_mean = self.z_mean_layer(x)
        z_log_var = self.z_log_var_layer(x)
        return z_mean, z_log_var

def variational_layer(z_mean, z_log_var, kl_loss_weight=1):
    '''
    잠재 변수의 샘플링
    Args:
        z_mean (Tensor): 잠재 변수의 평균
        z_log_var (Tensor): 잠재 변수의 로그 분산
        kl_loss_weight (float): KL 발산의 가중치 (기본값 1)
    Returns:
        Tensor: 샘플링된 잠재 변수
    '''
    return z_mean + torch.exp(0.5 * z_log_var) * torch.randn_like(z_log_var)

class RepeatVector(nn.Module):
    '''
    주어진 차원으로 벡터를 반복하는 레이어
    '''
    def __init__(self, n):
        super(RepeatVector, self).__init__()
        self.n = n

    def forward(self, x):
        '''
        입력 벡터를 주어진 차원으로 반복
        Args:
            x (Tensor): 입력 텐서
        Returns:
            Tensor: 반복된 텐서
        '''
        x = x.unsqueeze(1)
        x = x.repeat(1, self.n, 1)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decode_fc1 = nn.Linear(196, 196)
        self.decode_norm1 = nn.BatchNorm1d(196)
        dim = int(196)
        self.decode_fc2 = nn.Linear(196, dim)
        self.decode_norm2 = nn.BatchNorm1d(dim)
        self.repeat_vector = RepeatVector(120)
        self.gru = nn.GRU(196, 488, num_layers=3, batch_first=True)
        self.fc_out = nn.Linear(488, 35)

    def forward(self, z):
        '''
        디코더의 순방향 전파
        Args:
            z (Tensor): 잠재 변수
        Returns:
            Tensor: 재구성된 데이터의 확률 분포
        '''
        z = self.decode_norm1(torch.tanh(self.decode_fc1(z)))
        z = self.decode_norm2(torch.tanh(self.decode_fc2(z)))
        z = self.repeat_vector(z)
        z, _ = self.gru(z)
        z = self.fc_out(z)
        return F.softmax(z, dim=-1)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        '''
        VAE의 순방향 전파
        Args:
            x (Tensor): 입력 데이터
        Returns:
            tuple: 재구성된 데이터, 잠재 변수의 평균, 잠재 변수의 로그 분산
        '''
        z_mean, z_log_var = self.encoder(x)
        z = variational_layer(z_mean, z_log_var)
        reconstructed = self.decoder(z)
        return reconstructed, z_mean, z_log_var

    def generate(self, z):
        '''
        잠재 변수에서 샘플을 생성
        Args:
            z (Tensor): 잠재 변수
        Returns:
            Tensor: 생성된 데이터
        '''
        return self.decoder(z)

class PropertyPredictor(nn.Module):
    '''
    물리적 화학적 속성을 예측하는 신경망을 정의하는 클래스
    '''
    def __init__(self):
        super(PropertyPredictor, self).__init__()
        self.fc1 = nn.Linear(196, 67)
        self.fc2 = nn.Linear(67, 67)
        self.fc3 = nn.Linear(67, 67)
        self.output = nn.Linear(67, 1)
        self.dropout = nn.Dropout(0.15)

    def forward(self, x):
        '''
        속성 예측
        Args:
            x (Tensor): 입력 데이터
        Returns:
            Tensor: 예측된 속성
        '''
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.output(x)
        return x

def kl_div_loss(x_mean, x_log_var):
    '''
    KL 발산 손실을 계산
    Args:
        x_mean (Tensor): 잠재 변수의 평균
        x_log_var (Tensor): 잠재 변수의 로그 분산
    Returns:
        Tensor: KL 발산 손실
    '''
    kl_loss = -0.5 * torch.mean(1 + x_log_var - torch.pow(x_mean, 2) - torch.exp(x_log_var))
    return kl_loss
    
RDLogger.DisableLog('rdApp.*')

def is_valid_smiles(smiles_string):
    '''
    SMILES 문자열이 유효한지 확인하는 함수
    Args:
        smiles_string (str): SMILES 문자열
    Returns:
        bool: 유효한 SMILES 문자열이면 True 그렇지 않으면 False
    '''
    try:
        mol = Chem.MolFromSmiles(smiles_string)
        if mol is not None:
            return True
        else:
            return False
    except:
        return False

# 데이터로더를 정의
dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

model = VAE().to(device)
model_logP = PropertyPredictor().to(device)
model_qed = PropertyPredictor().to(device)
model_sas = PropertyPredictor().to(device)

# 옵티마이저 및 스케줄러를 정의
optimizer = optim.Adam(list(model.parameters()) + list(model_logP.parameters()) + list(model_qed.parameters()) + list(model_sas.parameters()), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=5, min_lr=0.00001)

# 손실 함수 정의
rc_loss = nn.BCELoss(reduction='sum')  
pr_loss = nn.MSELoss()  

# 학습 파라미터 및 기록 변수
NUM_EPOCHS = 200
BETA_MAX = 1.0
BETA_EPOCHS = 30

valid_smiles_ratio_train = []
valid_smiles_ratio_val = []

train_losses = []
train_rc_losses = []
train_kl_losses = []
train_pp_losses = []

val_losses = []
val_rc_losses = []
val_kl_losses = []
val_pp_losses = []

invalid_smiles_train = []
invalid_smiles_val = []


for epoch in range(NUM_EPOCHS):
    
    epoch_loss = 0
    epoch_rc_loss = 0
    epoch_kl_loss = 0
    epoch_pp_loss = 0

    cycle_epoch = epoch % (2 * BETA_EPOCHS)
    
    if cycle_epoch < BETA_EPOCHS:
        # Beta 값을 조정
        beta = BETA_MAX * torch.sigmoid(torch.tensor((cycle_epoch - BETA_EPOCHS / 2) / (BETA_EPOCHS / 10), dtype=torch.float32)).item()
    else:
        beta = 1.0
        
    print(f"================== Epoch -- {epoch} ==================")
    print(f"Current Beta: {beta:.5f}")
    
    model.train()
    for i, (data, properties) in enumerate(dataloader):
        train_input = data.float().to(device)
        properties = properties.float().to(device)
        
        # 속성을 분리
        true_logP, true_qed, true_sas = properties[:, 0], properties[:, 1], properties[:, 2]
        
        # 모델을 통해 재구성된 결과와 잠재 변수의 평균, 로그 분산을 얻음
        train_recon, z_mean, z_log_var = model(train_input)
        
        # 속성 예측을 수행
        pred_logP = model_logP(z_mean)
        pred_qed = model_qed(z_mean)
        pred_sas = model_sas(z_mean)

        # 각 속성에 대한 손실 계산
        log_loss = pr_loss(pred_logP.squeeze(), true_logP)
        qed_loss = pr_loss(pred_qed.squeeze(), true_qed)
        sas_loss = pr_loss(pred_sas.squeeze(), true_sas)
        
        # 재구성 손실, KL 발산 손실, 속성 예측 손실을 계산
        reconstruction_loss = rc_loss(train_recon, train_input) / BATCH_SIZE
        kl_loss = kl_div_loss(z_mean, z_log_var)
        pp_loss = log_loss + qed_loss + sas_loss
        
        # 전체 손실을 계산하고 역전파를 수행
        loss = reconstruction_loss + beta * kl_loss + beta * pp_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_rc_loss += reconstruction_loss.item()
        epoch_kl_loss += kl_loss.item()
        epoch_pp_loss += pp_loss.item()

    epoch_loss = epoch_loss / len(dataloader)
    epoch_rc_loss = epoch_rc_loss / len(dataloader)
    epoch_kl_loss = epoch_kl_loss / len(dataloader)
    epoch_pp_loss = epoch_pp_loss / len(dataloader)

    train_losses.append(epoch_loss)
    train_rc_losses.append(epoch_rc_loss)
    train_kl_losses.append(epoch_kl_loss)
    train_pp_losses.append(epoch_pp_loss)
    
    # 무작위 샘플을 출력
    data_point_sampled = random.randint(0, BATCH_SIZE - 1)
    print("Input  -- ", onehot_to_smiles(train_input[data_point_sampled].reshape(1, 120, len(vocab)).cpu().detach(), inv_dict))
    print("Output -- ", onehot_to_smiles(train_recon[data_point_sampled].reshape(1, 120, len(vocab)).cpu().detach(), inv_dict))
    
    print(f"Train LOSS: {epoch_loss:.5f} / RC LOSS: {epoch_rc_loss:.5f} / KL LOSS:{epoch_kl_loss:.5f} / PP LOSS:{epoch_pp_loss:.5f}")
    scheduler.step(epoch_loss)  # 학습률 조정
    
    # 검증 데이터로 평가
    model.eval()
    epoch_loss_val = 0
    epoch_rc_loss_val = 0
    epoch_kl_loss_val = 0
    epoch_pp_loss_val = 0
    with torch.no_grad():
        for i, (data, properties) in enumerate(val_dataloader):
            valid_input = data.float().to(device)
            properties = properties.float().to(device)
            
            true_logP, true_qed, true_sas = properties[:, 0], properties[:, 1], properties[:, 2]
            
            valid_recon, z_mean, z_log_var = model(valid_input)
            
            pred_logP = model_logP(z_mean)
            pred_qed = model_qed(z_mean)
            pred_sas = model_sas(z_mean)

            log_loss = pr_loss(pred_logP.squeeze(), true_logP)
            qed_loss = pr_loss(pred_qed.squeeze(), true_qed)
            sas_loss = pr_loss(pred_sas.squeeze(), true_sas)
            
            reconstruction_loss = rc_loss(valid_recon, valid_input) / BATCH_SIZE
            kl_loss = kl_div_loss(z_mean, z_log_var)
            pp_loss = log_loss + qed_loss + sas_loss
            
            loss = reconstruction_loss + beta * kl_loss + beta * pp_loss

            epoch_loss_val += loss.item()
            epoch_rc_loss_val += reconstruction_loss.item()
            epoch_kl_loss_val += kl_loss.item()
            epoch_pp_loss_val += pp_loss.item()
            
    # 검증 에포크 손실을 평균냄
    epoch_loss_val = epoch_loss_val / len(val_dataloader)
    epoch_rc_loss_val = epoch_rc_loss_val / len(val_dataloader)
    epoch_kl_loss_val = epoch_kl_loss_val / len(val_dataloader)
    epoch_pp_loss_val = epoch_pp_loss_val / len(val_dataloader)

    val_losses.append(epoch_loss_val)
    val_rc_losses.append(epoch_rc_loss_val)
    val_kl_losses.append(epoch_kl_loss_val)
    val_pp_losses.append(epoch_pp_loss_val)
    
    # 무작위 샘플을 출력
    data_point_sampled = random.randint(0, BATCH_SIZE - 1)
    print("Input  -- ", onehot_to_smiles(valid_input[data_point_sampled].reshape(1, 120, len(vocab)).cpu().detach(), inv_dict))
    print("Output -- ", onehot_to_smiles(valid_recon[data_point_sampled].reshape(1, 120, len(vocab)).cpu().detach(), inv_dict))
    
    print(f"Valid LOSS: {epoch_loss_val:.5f} / RC LOSS: {epoch_rc_loss_val:.5f} / KL LOSS:{epoch_kl_loss_val:.5f} / PP LOSS:{epoch_pp_loss_val:.5f}")
    print()

    if (epoch + 1) % 10 == 0:
        model.eval()
        
        # 학습 데이터에서 유효한 SMILES 비율을 계산
        valid_count_train = 0
        total_count_train = 0
        with torch.no_grad():
            for i, (data, properties) in enumerate(dataloader):
                train_input = data.float().to(device)
                train_recon, z_mean, z_log_var = model(train_input)
                
                for i in range(BATCH_SIZE):
                    smiles = onehot_to_smiles(train_recon[i].reshape(1, 120, len(vocab)).cpu().detach(), inv_dict)
                    if is_valid_smiles(smiles):
                        valid_count_train += 1
                    total_count_train += 1

        valid_ratio_train = (valid_count_train / total_count_train) * 100
        valid_smiles_ratio_train.append(valid_ratio_train)
        print(f"Train SMILES Total: {total_count_train}, Valid Train SMILES: {valid_count_train}, Valid Train SMILES Ratio: {valid_ratio_train:.2f}%")
        
        # 검증 데이터에서 유효한 SMILES 비율을 계산
        valid_count_val = 0
        total_count_val = 0
        with torch.no_grad():
            for i, (data, properties) in enumerate(val_dataloader):
                valid_input = data.float().to(device)
                valid_recon, z_mean, z_log_var = model(valid_input)
                
                for i in range(BATCH_SIZE):
                    smiles = onehot_to_smiles(valid_recon[i].reshape(1, 120, len(vocab)).cpu().detach(), inv_dict)
                    if is_valid_smiles(smiles):
                        valid_count_val += 1
                    total_count_val += 1

        valid_ratio_val = (valid_count_val / total_count_val) * 100
        valid_smiles_ratio_val.append(valid_ratio_val)
        print(f"Valid SMILES Total: {total_count_val}, Valid SMILES: {valid_count_val}, Valid SMILES Ratio: {valid_ratio_val:.2f}%")

        # 학습 데이터와 검증 데이터에서 유효하지 않은 SMILES를 저장
        with torch.no_grad():
            for i, (data, properties) in enumerate(dataloader):
                train_input = data.float().to(device)
                train_recon, z_mean, z_log_var = model(train_input)
                
                for i in range(BATCH_SIZE):
                    smiles = onehot_to_smiles(train_recon[i].reshape(1, 120, len(vocab)).cpu().detach(), inv_dict)
                    if not is_valid_smiles(smiles) and len(invalid_smiles_train) < 100:
                        invalid_smiles_train.append(smiles)

        with torch.no_grad():
            for i, (data, properties) in enumerate(val_dataloader):
                valid_input = data.float().to(device)
                valid_recon, z_mean, z_log_var = model(valid_input)
                
                for i in range(BATCH_SIZE):
                    smiles = onehot_to_smiles(valid_recon[i].reshape(1, 120, len(vocab)).cpu().detach(), inv_dict)
                    if not is_valid_smiles(smiles) and len(invalid_smiles_val) < 100:
                        invalid_smiles_val.append(smiles)

        # 유효하지 않은 SMILES를 파일로 저장
        with open('invalid_smiles_train.txt', 'w') as f:
            for smiles in invalid_smiles_train:
                f.write(smiles + '\n')
        
        with open('invalid_smiles_val.txt', 'w') as f:
            for smiles in invalid_smiles_val:
                f.write(smiles + '\n')

    print()


print("\n===================================== Save model =====================================\n")
save_dict = {
    'model_state_dict': model.state_dict(),
    'model_logP_state_dict': model_logP.state_dict(),
    'model_qed_state_dict': model_qed.state_dict(),
    'model_sas_state_dict': model_sas.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_losses': train_losses,
    'train_rc_losses': train_rc_losses,
    'train_kl_losses': train_kl_losses,
    'train_pp_losses': train_pp_losses,
    'val_losses': val_losses,
    'val_rc_losses': val_rc_losses,
    'val_kl_losses': val_kl_losses,
    'val_pp_losses': val_pp_losses,
    'valid_smiles_ratio_train': valid_smiles_ratio_train,
    'valid_smiles_ratio_val': valid_smiles_ratio_val,
    'vocab': vocab,
    'inv_dict': inv_dict,
    'epoch': epoch
}

torch.save(save_dict, f"{fname}_model.pth")


print("\n========================= PCA Plot =============================\n")

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

model.eval()
with torch.no_grad():
    z_means_train = []
    true_properties_train = []
    
    # 검증 데이터로부터 잠재 변수와 실제 속성을 수집
    for data, properties in val_dataloader:
        data = data.float().to(device)
        z_mean, _ = model.encoder(data)
        z_means_train.append(z_mean.cpu().numpy())
        true_properties_train.append(properties.cpu().numpy())

    # 리스트를 배열로 변환
    z_means_train = np.vstack(z_means_train)
    true_properties_train = np.vstack(true_properties_train)

# 실제 속성 값에서 QED, SAS를 추출하고, GP 값을 계산
qed_values = true_properties_train[:, 1]
sas_values = true_properties_train[:, 2]
gp_values = 5 * qed_values - sas_values

# PCA를 사용하여 잠재 공간을 2차원으로 축소
pca = PCA(n_components=2)

z_pca_logP = pca.fit_transform(z_means_train)
z_pca_qed = pca.fit_transform(z_means_train)
z_pca_sas = pca.fit_transform(z_means_train)
z_pca_gp = pca.fit_transform(z_means_train)

# PCA 결과를 시각화
plt.figure(figsize=(20, 16))

plt.subplot(2, 2, 1)
plt.scatter(z_pca_logP[:, 0], z_pca_logP[:, 1], c=true_properties_train[:, 0], cmap='viridis', marker='.')
plt.colorbar()
plt.title('PCA on Latent Space - logP')

plt.subplot(2, 2, 2)
plt.scatter(z_pca_qed[:, 0], z_pca_qed[:, 1], c=true_properties_train[:, 1], cmap='viridis', marker='.')
plt.colorbar()
plt.title('PCA on Latent Space - QED')

plt.subplot(2, 2, 3)
plt.scatter(z_pca_sas[:, 0], z_pca_sas[:, 1], c=true_properties_train[:, 2], cmap='viridis', marker='.')
plt.colorbar()
plt.title('PCA on Latent Space - SAS')

plt.subplot(2, 2, 4)
plt.scatter(z_pca_gp[:, 0], z_pca_gp[:, 1], c=gp_values, cmap='viridis', marker='.')
plt.colorbar()
plt.title('PCA on Latent Space - GP (5 * QED - SAS)')

plt.savefig(f"{fname}_property.png")

print("\n========================= Loss graph =============================\n")

# loss 그래프를 시각화
plt.figure(figsize=(12, 8))
plt.plot(train_losses, label='Train Total Loss')
plt.plot(val_losses, label='Validation Total Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Total Loss per Epoch')
plt.legend()
plt.savefig(f"{fname}_Totalloss.png")

plt.figure(figsize=(12, 8))
plt.plot(train_rc_losses, label='Train Reconstruction Loss')
plt.plot(val_rc_losses, label='Validation Reconstruction Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title(' Reconstruction Loss per Epoch')
plt.legend()
plt.savefig(f"{fname}_Reconloss.png")

plt.figure(figsize=(12, 8))
plt.plot(train_kl_losses, label='Train KL Loss')
plt.plot(val_kl_losses, label='Validation KL Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title(' KL Loss per Epoch')
plt.legend()
plt.savefig(f"{fname}_KLloss.png")

plt.figure(figsize=(12, 8))
plt.plot(train_pp_losses, label='Train Property Loss')
plt.plot(val_pp_losses, label='Validation Property Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title(' PropertyLoss per Epoch')
plt.legend()
plt.savefig(f"{fname}_propertyloss.png")


print("\n========================= Accuracy graph =============================\n")


# 에포크에 따른 유효한 SMILES 비율을 시각화
epochs = list(range(10, NUM_EPOCHS + 1, 10))

plt.figure(figsize=(14, 6))

plt.plot(epochs, valid_smiles_ratio_train, label='Train Valid Molecule %')
plt.plot(epochs, valid_smiles_ratio_val, label='Validation Valid Molecule %')
plt.xlabel('Epochs')
plt.ylabel('Valid Molecule (%)')
plt.title('Valid Molecule Percentage over Epochs')
plt.legend()

plt.tight_layout()
plt.savefig(f"{fname}_Accuracy.png")

print("\n============================ Interpolation =================================\n") 

# 샘플 SMILES를 정의
pfizer = 'CCn1cncc1Cn1c(CN2CCC(c3cccc(OCc4ccc(F)cc4F)n3)CC2)nc2ccc(C(=O)O)cc21'
test_data = make_one_hot([pfizer], vocab)

model.eval()

test_input = torch.tensor(test_data, dtype=torch.float32).to(device)

with torch.no_grad():
    test_recon, z_mean, z_log_var = model(test_input)

# 거리 값을 정의하고 다양한 거리에서 SMILES를 샘플링
distances = np.arange(0.1, 2.1, 0.2)
generated_smiles_list = []

for distance in distances:
    print(f"Sampling at distance: {distance}")
    
    valid_smiles_found = False
    num_samples = 0 
    max_samples = 100000
    
    while not valid_smiles_found and num_samples < max_samples:
        num_samples += 1
        
        # 잠재 공간에서 샘플링하여 새로운 SMILES를 생성
        z_sampled = z_mean + distance * torch.randn_like(z_mean)
        
        with torch.no_grad():
            generated_recon = model.generate(z_sampled)
            generated_smiles = onehot_to_smiles(generated_recon.cpu().detach().numpy(), inv_dict)

        mol = Chem.MolFromSmiles(generated_smiles)
        if mol is not None:
            valid_smiles_found = True
            generated_smiles_list.append((distance, generated_smiles))
    
    if valid_smiles_found:
        print(f"Valid molecule found at distance {distance}: {generated_smiles}")
    else:
        print(f"No valid molecule found at distance {distance} after {max_samples} attempts.")

print("\nGenerated SMILES Samples at Different Distances:")
for distance, smiles in generated_smiles_list:
    print(f"{smiles} : Distance {distance:.2f}")

# 더 긴 거리 범위에서 SMILES를 샘플링
distances = np.arange(3, 21, 3)
generated_smiles_list = []

for distance in distances:
    print(f"Sampling at distance: {distance}")
    
    valid_smiles_for_distance = []
    num_samples = 0 
    max_samples = 1000 
    
    while len(valid_smiles_for_distance) < 10 and num_samples < max_samples:
        num_samples += 1
        
        z_sampled = z_mean + distance * torch.randn_like(z_mean)
        
        with torch.no_grad():
            generated_recon = model.generate(z_sampled)
            generated_smiles = onehot_to_smiles(generated_recon.cpu().detach().numpy(), inv_dict)

        mol = Chem.MolFromSmiles(generated_smiles)
        if mol is not None:
            valid_smiles_for_distance.append(generated_smiles)
    
    generated_smiles_list.append((distance, valid_smiles_for_distance))
    print(f"Valid molecules found at distance {distance}: {len(valid_smiles_for_distance)}")
    for smiles in valid_smiles_for_distance:
        print(smiles)

print("\nGenerated SMILES Samples at Different Distances:")
for distance, smiles_list in generated_smiles_list:
    print(f"Distance {distance:.2f}:")
    for smiles in smiles_list:
        print(smiles)


print("\n===================================== Optimization =====================================\n")

import torch.optim as optim
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize
import numpy as np
import pandas as pd
import math
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.QED import qed
import torch
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.optimize import minimize
from rdkit import Chem

def sample_latent_vector(z_mean, z_log_var):
    # 잠재 벡터를 샘플링
    epsilon = torch.randn_like(z_log_var)  # z_log_var와 같은 형태의 표준 정규분포 난수를 생성
    return z_mean + torch.exp(0.5 * z_log_var) * epsilon  # 샘플링된 벡터를 반환

start_smiles = 'C=CCN(Cc1cc(=O)n2cccc(C)c2[nH+]1)Cc1cccs1'

percentile_df = pd.read_csv('percentile.csv')

def get_properties_from_csv(smiles):
    # SMILES 문자열로부터 속성을 가져오기
    entry = percentile_df[percentile_df['smiles'] == smiles] 
    if entry.empty:
        raise ValueError(f"SMILES not found in CSV: {smiles}")  # SMILES가 CSV에 없으면 예외를 발생
    # QED, SAS, Percentile 값을 추출
    qed_value = entry['qed'].values[0]
    sas_value = entry['SAS'].values[0]
    percentile = entry['percentile'].values[0]
    return qed_value, sas_value, percentile

try:
    # 시작 SMILES의 속성을 가져오기
    start_qed, start_sas, start_percentile = get_properties_from_csv(start_smiles)
    print(f"Start SMILES: {start_smiles}, QED: {start_qed:.2f}, SAS: {start_sas:.2f}, Percentile: {start_percentile:.2f}%")
except ValueError as e:
    print(e) 

z_train = []
y_train = []

with torch.no_grad():
    # 학습 데이터로부터 잠재 변수와 속성을 추출
    for data, properties in dataloader:
        data = data.float().to(device) 
        z_mean, z_log_var = model.encoder(data)  # 데이터로부터 z_mean과 z_log_var를 인코딩
        # 잠재 벡터를 샘플링하여 z_train 리스트에 추가
        z_train.append(sample_latent_vector(z_mean.cpu(), z_log_var.cpu()))
        # 속성을 계산하여 y_train 리스트에 추가
        y_train.append(5 * properties[:, 1] - properties[:, 2])

# z_train과 y_train 리스트를 배열로 변환
z_train = np.concatenate(z_train, axis=0)
y_train = np.concatenate(y_train, axis=0)

sample_size = 10000
# 전체 z_train에서 샘플 크기만큼 무작위로 샘플을 선택
sample_indices = np.random.choice(len(z_train), sample_size, replace=False)

# 선택된 샘플 인덱스를 사용하여 z_train_sampled와 y_train_sampled를 생성
z_train_sampled = z_train[sample_indices]
y_train_sampled = y_train[sample_indices]

# Gaussian Process 모델을 훈련
gp_model = GaussianProcessRegressor()
gp_model.fit(z_train_sampled, y_train_sampled)  # 훈련 데이터로 Gaussian Process 모델을 적합

# 시작 SMILES를 잠재 공간으로 인코딩
start_mol = Chem.MolFromSmiles(start_smiles)  # SMILES 문자열을 RDKit Mol 객체로 변환
with torch.no_grad():
    start_input = make_one_hot([start_smiles], vocab).astype(np.float32)  # SMILES를 one-hot 인코딩으로 변환
    start_input = torch.tensor(start_input).to(device) 
    
    start_input_repeated = start_input.repeat(10, 1, 1) 
    z_mean, z_log_var = model.encoder(start_input_repeated)  # 인코더를 통해 z_mean과 z_log_var를 계산
    
    # 첫 번째 배치의 z_mean과 z_log_var를 추출.
    z_mean = z_mean[0].unsqueeze(0)
    z_log_var = z_log_var[0].unsqueeze(0)

# GP 모델을 사용하여 잠재 벡터를 최적화
z_init = sample_latent_vector(z_mean, z_log_var).cpu().numpy().flatten()  # 초기 잠재 벡터를 샘플링하고 numpy 배열로 변환합니다.

def objective(z):
    # 최적화 목표 함수입니다. GP 모델을 사용하여 예측값의 음수를 반환합니다.
    return -gp_model.predict(z.reshape(1, -1))  # 예측값의 음수를 반환하여 최적화 목표를 설정합니다.

# 최적화를 수행
res = minimize(objective, z_init, method='L-BFGS-B', options={'maxiter': 200})

# 최적화된 z 값을 추출
z_optimized = res.x

# z_init과 z_optimized 사이의 interpolation
n_steps = 20
# interpolation 경로
z_interp = [z_init + (z_optimized - z_init) * t / n_steps for t in range(n_steps + 1)]

# 경로에 따른 분자의 속성을 계산
smiles_list = []
properties_list = []

model.eval() 

for z in z_interp:
    z_tensor = torch.tensor(z).unsqueeze(0).float().to(device)  .
    generated_molecule = model.decoder(z_tensor)  # 디코더를 통해 분자를 생성
    generated_smiles = onehot_to_smiles(generated_molecule.cpu().detach(), inv_dict)  # 생성된 분자를 SMILES 문자열로 변환
    
    try:
        # 생성된 SMILES의 속성을 가져오기
        qed_value, sas_value, percentile = get_properties_from_csv(generated_smiles)
        smiles_list.append(generated_smiles)
        properties_list.append((qed_value, sas_value, percentile))
    except ValueError as e:
        print(e) 
        smiles_list.append(None)
        properties_list.append((None, None, None))

for i, (smiles, (qed, sas, percentile)) in enumerate(zip(smiles_list, properties_list)):
    if smiles is not None:
        label = "Finish" if i == len(smiles_list) - 1 else ("Start" if i == 0 else f"Step {i}")
        print(f"{label}: SMILES: {smiles}, QED: {qed:.2f}, SAS: {sas:.2f}, Percentile: {percentile:.2f}%")
    else:
        print(f"Step {i}: Generated SMILES is invalid.")  # 유효하지 않은 SMILES가 생성된 경우 메시지를 출력합니다.


from rdkit.Chem import Draw
mol_images = []
if start_mol:
    start_img = Draw.MolToImage(start_mol, size=(200, 200))
    mol_images.append(start_img)
    for smiles in smiles_list:
        if smiles is not None:
            mol = Chem.MolFromSmiles(smiles)
            img = Draw.MolToImage(mol, size=(200, 200))
            mol_images.append(img)

# mol_images가 비어 있지 않은지 확인한 후 시각화
if mol_images:
    fig, axes = plt.subplots(1, len(mol_images), figsize=(15, 5))

    # 서브플롯이 하나만 있는 경우에도 axes를 리스트로 변환
    if len(mol_images) == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.imshow(mol_images[i])
        if i == 0:
            ax.set_title(f"Start\n({start_qed:.2f},{start_sas:.2f},{start_percentile:.2f}%)")
        elif properties_list[i-1][0] is not None:
            ax.set_title(f"({properties_list[i-1][0]:.2f},{properties_list[i-1][1]:.2f},{properties_list[i-1][2]:.2f}%)")
        ax.axis('off')

    plt.tight_layout()
    plt.show()
else:
    print("No valid molecules were generated.")

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import torch

# z_init과 z_optimized가 numpy 배열이라고 가정
z_init_np = np.array(z_init).reshape(1, -1)
z_optimized_np = np.array(z_optimized).reshape(1, -1)

model.eval()

# 이전 섹션에서 z_means_train을 가져오기
with torch.no_grad():
    z_means_train = []
    true_properties_train = []
    
    for data, properties in val_dataloader:
        data = data.float().to(device)
        z_mean, _ = model.encoder(data)
        z_means_train.append(z_mean.cpu().numpy())
        true_properties_train.append(properties.cpu().numpy())

    z_means_train = np.vstack(z_means_train)
    true_properties_train = np.vstack(true_properties_train)

# GP 값 계산
qed_values = true_properties_train[:, 1]
sas_values = true_properties_train[:, 2]
gp_values = 5 * qed_values - sas_values

# PCA 모델 생성
pca = PCA(n_components=2)

# 훈련된 잠재 공간에 PCA를 적합
z_pca = pca.fit_transform(z_means_train)

# z_init과 z_optimized를 PCA로 변환
z_init_pca = pca.transform(z_init_np)
z_optimized_pca = pca.transform(z_optimized_np)

# PCA 결과를 시각화
plt.figure(figsize=(10, 8))
plt.scatter(z_pca[:, 0], z_pca[:, 1], c=gp_values, cmap='viridis', marker='.')
plt.colorbar(label='GP Value (5 * QED - SAS)')

# z_init과 z_optimized의 포인트를 플롯
plt.scatter(z_init_pca[:, 0], z_init_pca[:, 1], c='red', label='z_init', edgecolor='black')
plt.scatter(z_optimized_pca[:, 0], z_optimized_pca[:, 1], c='blue', label='z_optimized', edgecolor='black')

# z_init과 z_optimized 사이에 선을 그리기
plt.plot([z_init_pca[0, 0], z_optimized_pca[0, 0]], 
         [z_init_pca[0, 1], z_optimized_pca[0, 1]], 
         'k--', linestyle='--', linewidth=1)

plt.legend()
plt.title('PCA on Latent Space with z_init and z_optimized')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()
