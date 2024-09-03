import torch, dgl, pickle
from rdkit.Avalon import pyAvalonTools
import pandas as pd
from glob import glob
from rdkit import Chem
from rdkit.Chem import AllChem
import os, random
import dgl, torch
import pandas as pd
from glob import glob
from time import time
from tqdm import tqdm
from dgl.data.utils import Subset, save_graphs, load_graphs
from dgl.dataloading import GraphDataLoader, DataLoader
from model import PredictionScore
from train import run_train_epoch, run_eval_epoch
from data import ScoreDataset
from rdkit.Chem.rdReducedGraphs import GetErGFingerprint

df = pd.read_csv('./modi_46smiles.csv')

# 화합물 파일명, SMILES, 그리고 결합 에너지를 이용해 딕셔너리를 생성
dict = {}
for file, smi, score in tqdm(zip(df['FileName'], df['SMILES'], df['BindingEnergy'])):
    try:
        # SMILES 문자열을 이용해 Molecule 객체를 생성하고 이를 딕셔너리에 저장
        dict[file[:-4]] = (Chem.MolFromSmiles(smi), score)
    except:
        pass

# 생성한 딕셔너리를 피클 파일로 저장
with open('data.pickle', 'wb') as F:
    pickle.dump(dict, F)

# 분자의 fingerprint를 얻기
def get_fps(mol):
    # Morgan, Avalon, ErG fingerprint를 계산하고 torch 텐서로 변환
    morgan = torch.tensor(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)).float()
    avalon = torch.tensor(pyAvalonTools.GetAvalonFP(mol)).float()
    erg = torch.tensor(GetErGFingerprint(mol)).float()

    return morgan, avalon, erg

# atom의 특성을 원핫 인코딩으로 변환
def one_hot(x, allowable_set):
    if x not in allowable_set:            
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

# 원자, 결합에 대한 허용 가능한 특성 값들을 정의
allow_symbol  = ['C', 'O', 'N', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'X']
allow_degree  = [i for i in range(7)]
allow_totalHs = [i for i in range(5)]
allow_hybrid  = [Chem.rdchem.HybridizationType.SP, 
                 Chem.rdchem.HybridizationType.SP2, 
                 Chem.rdchem.HybridizationType.SP3, 
                 Chem.rdchem.HybridizationType.SP3D, 
                 Chem.rdchem.HybridizationType.SP3D2, 
                 Chem.rdchem.HybridizationType.UNSPECIFIED]
allow_bond = [Chem.rdchem.BondType.SINGLE, 
              Chem.rdchem.BondType.DOUBLE, 
              Chem.rdchem.BondType.TRIPLE, 
              Chem.rdchem.BondType.AROMATIC]
allow_streo = [Chem.rdchem.BondStereo.STEREOANY,
               Chem.rdchem.BondStereo.STEREOCIS,
               Chem.rdchem.BondStereo.STEREOE,
               Chem.rdchem.BondStereo.STEREONONE,
               Chem.rdchem.BondStereo.STEREOTRANS,
               Chem.rdchem.BondStereo.STEREOZ] 

# 주어진 원자의 특성을 원핫 벡터로 변환
def atom_feature(atom):
    symbol   = one_hot(atom.GetSymbol(), allow_symbol)
    degree   = one_hot(atom.GetDegree(), allow_degree)
    total_H  = one_hot(atom.GetTotalNumHs(), allow_totalHs)              
    hybrid   = one_hot(atom.GetHybridization(), allow_hybrid)
    aromatic = [atom.GetIsAromatic()]
    isinring = [atom.IsInRing()]
    formal_charge = [atom.GetFormalCharge() * 0.2]
    return symbol + degree + total_H + hybrid + aromatic + isinring + formal_charge 

# 주어진 결합의 특성을 원핫 벡터로 변환
def bond_feature(bond):
    bond_type  = one_hot(bond.GetBondType(), allow_bond)
    bond_streo = one_hot(bond.GetStereo(), allow_streo)
    isinring   = [bond.IsInRing()]
    conjugated = [bond.GetIsConjugated()]
    return bond_type + bond_streo + isinring + conjugated

# 분자의 원자 특성을 계산
def get_atom_feature(mol): 
    return torch.tensor([atom_feature(atom) for atom in mol.GetAtoms()]).float()

# 분자의 결합 특성을 계산
def get_bond_feature(mol, u, v):
    adj = torch.tensor(Chem.GetAdjacencyMatrix(mol))

    edge_feat = []
    for src, dst in zip(u, v):
        bf = bond_feature(mol.GetBondBetweenAtoms(int(src), int(dst)))
        edge_feat.append(bf)    
    return torch.tensor(edge_feat).float()

# 각 분자에 대해 특성을 계산하고 그래프를 생성하여 저장
for idx, (file, (mol, score)) in enumerate(dict.items()):
    # 분자의 fingerprint를 계산
    morgan, avalon, erg = get_fps(mol)
    
    # 인접 행렬을 이용해 그래프를 생성
    edge = torch.tensor(Chem.GetAdjacencyMatrix(mol)).to_sparse()
    u, v = edge.indices()
    
    g = dgl.graph((u, v))
    nf = get_atom_feature(mol)
    ef = get_bond_feature(mol, u, v)

    g.ndata['feat'] = nf
    g.edata['feat'] = ef

    # 그래프와 레이블을 파일로 저장
    labels = {'morgan': morgan, 'avalon': avalon, 'erg': erg, 'score': torch.tensor([score]).float()}
    
    save_graphs(f'./46data/{file}.bin', g, labels)

# 생성된 데이터셋을 로드하고, 훈련 및 검증 데이터로 나눔
train = ScoreDataset(sorted(glob("./46data/*")))

train_range = range(len(train))
valid_index = random.sample(train_range, 1000)
train_index = list(set(train_range) - set(valid_index))

train_data = Subset(train, train_index)
valid_data = Subset(train, valid_index)

# DataLoader를 설정
batch_size = 128
train_loader = GraphDataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=True)
valid_loader = GraphDataLoader(valid_data, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=True)

# 모델 초기화 및 학습 설정
device = 'cuda'
model = PredictionScore(in_size=31, emb_size=256, edge_size=12, num_layers=4, dropout_ratio=0.2).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

train_losses = []
valid_losses = []
corr = []

for epoch in range(1, 50 + 1):
    print("-" * 100)
    
    train_loss = run_train_epoch(model, train_loader, optimizer, device=device)
    valid_loss, valid_true, valid_pred  = run_eval_epoch(model, valid_loader, device=device)
    valid_corr = torch.corrcoef(torch.stack([valid_true, valid_pred]))[0][1]
    
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    corr.append(valid_corr)
    
    print(f'Train Loss: {train_loss:.3f},  Valid Loss: {valid_loss:.3f},  Valid Corr: {valid_corr:.3f}')

import matplotlib.pyplot as plt

# 실제값 vs. 예측값 scatter plot
plt.figure(figsize=(8, 8))
plt.scatter(valid_true, valid_pred, color='blue', alpha=0.2)
plt.plot([min(valid_true), max(valid_true)], [min(valid_true), max(valid_true)], color='red', linestyle='--')  
plt.xlabel('True ADG score')
plt.ylabel('Predicted ADG score')
plt.title('True vs. Predicted ADG score')

plt.grid(True)
plt.xlim(-12, -2.5)

plt.savefig("truevspredicted.png", dpi=300, bbox_inches='tight')

# loss graph plot
epochs = range(1, len(train_losses) + 1)
plt.plot(epochs, train_losses, color='red', label='Training loss')
plt.plot(epochs, valid_losses, color='green', label='Validation loss')
plt.plot(epochs, corr, color='blue', label='correlation')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig("losss.png", dpi=300, bbox_inches='tight')

# residual plot
plt.figure(figsize=(8, 8))
error = valid_pred - valid_true
plt.scatter(valid_true, error, color='blue', alpha=0.2)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('True ADG score')
plt.ylabel('Error')
plt.title('Prediction error plot')
plt.grid(True)
plt.ylim(-1, 1.5)
plt.savefig("error.png", dpi=300, bbox_inches='tight')
