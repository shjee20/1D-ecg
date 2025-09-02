
본 저장소는 「딥러닝 기반 다중라벨 12리드 심전도 데이터 분류 성능 평가」의 실험 코드를 포함하고 있습니다.  
PhysioNet 2021 Challenge ECG 데이터를 기반으로 1D ResNet 모델을 이용한 다중라벨 심전도 분류기를 구현하고, 리드 수, 전처리 방식, threshold 조정이 성능에 미치는 영향을 분석하였습니다.



## 주요 특징
- 데이터셋: PhysioNet CinC Challenge 2021 (88,253 ECG recordings, 23 classes 사용)  
- 모델 구조: 1D ResNet 기반 CNN (ResNet-50 변형 구조)  
- 멀티라벨 분류: 23개 심전도 질환 클래스, 샘플당 최대 8개 라벨  
- 전처리 방식:  
  - Raw (bandpass filter + 표준화)  
  - Pan–Tompkins (QRS 강조, R-R 간격 기반)  
  - DWT (Discrete Wavelet Transform, baseline drift 제거)  
- 리드 수 비교: 12, 6, 4, 3, 2 leads  
- Threshold 전략:  
  - Fixed threshold (0.5)  
  - Variable threshold (클래스별 AUROC 기반 최적화, Youden index)  




## 코드 구성

```
.
├── train.ipynb        # 학습 실행용 노트북
├── test.ipynb         # 테스트 및 평가 실행용 노트북
├── team_code.py       # 학습/검증/테스트 전체 pipeline
├── helper_code.py     # 데이터 로딩, 전처리, 커스텀 metric 및 callback
├── model_type/ResNet  # ResNet-1D 모델 정의
└── README.md          # 프로젝트 설명 파일 (본 문서)
```



## 주요 파일 설명

### team_code.py
- train_model, train_model_2: ECG 데이터 학습 함수
- run_model: 학습된 모델 로드 후 테스트 데이터 예측
- generate_X_rawecg, generate_X_dwt, generate_X_pan_tompkins: 리드 수/전처리별 입력 생성기
- Keras 기반 학습/로그 저장/threshold 계산 callback 포함

### helper_code.py
- 데이터 입출력: find_challenge_files, load_challenge_data
- 신호 처리: bandpass_filter, perform_discrete_wavelet_transform, pan_tompkins
- 메트릭: CustomF1WithClassThresholds, DynamicF1
- 콜백: ROCThresholdExcelCallback, NaNLossRecoveryCallback

### train.ipynb
- PhysioNet 데이터 로딩 및 전처리
- 모델 학습 실행
- 로그 및 F1-score 기록

### test.ipynb
- 학습된 모델 불러오기
- 클래스별 threshold 적용 후 테스트셋 평가
- 결과 시각화 및 ROC/F1-score 계산


## 설치 & 실행 방법

### 1. 라이브러리 설치
```bash
pip install numpy pandas scikit-learn seaborn matplotlib tensorflow tensorflow-addons scipy joblib tqdm pywt
```

### 2. 데이터 준비
- PhysioNet CinC 2021 ECG 데이터 다운로드
' https://moody-challenge.physionet.org/2021/ '
- .mat, .hea 파일을 프로젝트 데이터 폴더에 배치

### 3. 학습 실행
```bash
jupyter notebook train.ipynb
```

### 4. 테스트 실행
```bash
jupyter notebook test.ipynb
```





