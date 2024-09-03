# ADG
이 프로젝트는 Docking 시뮬레이션을 수행하고, 그 결과를 기반으로 딥러닝을 통해 스코어를 예측하는 workflow이다.

1. docking.sh: AutoDock GPU를 사용하여 도킹 시뮬레이션을 수행하는 셸 스크립트
2. job.py: 작업 관리 및 실행을 위한 스크립트

3. predictor directory

- data.py

그래프 데이터 로딩 및 처리를 위한 ScoreDataset 클래스 정의

ScoreDataset: DGLDataset를 상속받아 그래프와 관련된 특성 및 레이블을 로드하는 메소드 제공

- model.py
  
모델 아키텍처를 정의하며, Multihead attention mechanism과 그래프 변환 모듈을 포함

MultiHeadAttentionLayer: 그래프의 노드와 엣지에 대해 멀티헤드 어텐션 구현

GraphTransformerModule: 그래프 변환 레이어를 적용하여 노드와 엣지의 특징을 업데이트

PredictionScore: 그래프 변환 모듈과 핑거프린트 특징을 결합하여 점수를 예측하는 모델 정의


- train.py
  
모델의 훈련 및 평가 프로세스 관리

run_train_epoch: 모델을 한 에폭 동안 훈련

run_eval_epoch: 검증 데이터셋에서 모델 평가


- predictor.py

분자 데이터를 처리하고 그래프를 생성하며, 모델을 훈련하고 평가

데이터 준비: SMILES 문자열을 분자 그래프로 변환하고 다양한 핑거프린트를 계산

모델 훈련: 데이터 로더를 설정하고 모델을 초기화하여 여러 에폭 동안 훈련

평가:plot을 계산하고 저장


# ChemicalVAE

이 프로젝트는 SMILES 데이터를 Input으로 하는 Variational Autoencoder, VAE 모델을 구현해 분자를 생성하고 생성된 분자들의 화학적 특성  최적화를 가능하게 한다.


분자 데이터 처리: SMILES 문자열을 토큰화하고 원-핫 인코딩 벡터로 변환

VAE: 분자의 표현을 학습

속성 예측: 분자의 logP, QED, SAS 값을 예측

평가: 생성된 SMILES 문자열의 유효성을 평가

