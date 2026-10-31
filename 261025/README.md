# 261025 과제 - 딥러닝 및 GPU 가속

`03_DL-and-GPU.ipynb`의 모델 성능 확인부터 CuPy convolution 연산까지의 TODO를 구현한 과제입니다.

앞에서 배운 CNN 학습 코드와 Feature Map 시각화 함수를 사용하고, NumPy와 CuPy로 이미지 잔상 효과와 convolution을 구현하여 CPU와 GPU의 결과 및 처리 시간을 비교하는 과제를 수행하였습니다.

<p align="center">
  <img src/pictures="CNN_Loss.jpeg" width="25%">
  <img src/pictures="CNN_Accuracy.jpeg" width="25%">
  <img src/pictures="Comparison1.jpeg" width="25%">
  <img src/pictures="Comparison2.jpeg" width="25%">
</p>

## 1. 파일 구조와 역할

```text
261025/
├── 261025_task.py       # 실습 준비 코드와 TODO 풀이
├── README.md            # 이 문서
└── src/
    └── images/
        └── seagull.jpg  # 입력 이미지
```

- `261025_task.py`에는 필요한 import, 함수, 모델 정의, 데이터 로딩과 학습 코드가 들어 있습니다.
- MNIST 데이터는 최초 실행 시 `src/datasets`에 다운로드됩니다.
- `src/images/seagull.jpg` 이미지 파일도 폴더에 넣어서, 261025 폴더 만으로도 파일을 실행할 수 있도록 만들었습니다.

## 2. 동작 과정

### 모델 성능 확인

- 기존 `SimpleCNN`, `train_one_epoch()`, `evaluate()`를 사용해 MNIST를 CPU에서 3 Epoch 학습하고 평가합니다.
- 각 Epoch의 Train/Test Loss와 Accuracy를 저장하고, 학습 결과와 소요 시간을 출력합니다.
- 저장한 기록으로 Loss와 Accuracy 그래프를 출력합니다.
- Accuracy는 NumPy 배열로 변환한 뒤 100을 곱해 백분율로 표시합니다.

### 학습된 모델의 Feature Map 확인

- 테스트 이미지 한 장을 선택하고 Batch 차원을 추가합니다.
- `model.eval()`과 `torch.no_grad()`를 사용해 평가 모드에서 Gradient 계산 없이 순전파합니다.
- `return_features=True`로 각 층의 중간 출력을 가져옵니다.
- 기존 `show_feature_maps()`를 사용해 Conv1은 8개, Conv2는 16개를 출력합니다.
- conv1, relu1, pool1, conv2, relu2, pool2는 각각 4개씩 출력하여 비교합니다.
- 색상은 `viridis`를 사용합니다.

### 이미지 잔상 효과 구현

- 갈매기 이미지를 흑백으로 불러오고 1280×720 크기로 조정합니다.
- 픽셀값을 float32의 0~1 범위로 변환합니다.
- 오른쪽으로 이동할수록 작아지는 가중치 32개를 만들고, 합이 1이 되도록 정규화합니다.
- NumPy에서는 출력 픽셀을 하나씩 계산합니다.
- CuPy에서는 이동량만 반복하고, 각 이동 단계의 픽셀 계산은 배열 전체에 적용합니다.
- CPU 처리 시간, GPU 처리 시간, CPU와 GPU 사이의 복사 시간을 출력합니다.
- 복사 시간을 제외한 가속 배율과 포함한 가속 배율을 각각 구합니다.
- 원본, CPU 잔상, GPU 잔상 이미지를 출력하여 비교합니다.

### Convolution 연산

- 실습에서 사용한 가로·세로 edge 필터와 편향을 사용합니다.
- CPU에서는 기존 `convolution2d()`를 사용합니다.
- GPU에서는 커널 원소를 하나씩 선택하고, 대응하는 입력 영역 전체에 가중치를 곱해 누적합니다.
- 0으로 채운 배열에 입력 이미지를 넣어서 zero padding을 구성합니다.
- 두 구현 모두 `stride=1`, `padding=1`로 계산합니다.
- GPU 결과를 `cp.asnumpy()`로 복사한 뒤 기존 `relu()`를 적용합니다.
- CPU/GPU 처리 시간과 가속 배율을 출력합니다.
- ReLU를 적용한 결과끼리 최대 절대 차이를 구하고, 가로·세로 edge 이미지를 비교합니다.

## 3. 실행 방법

`261025_task.py`가 있는 폴더 안에서 실행합니다.

```bash
python 261025_task.py
```

- 입력 이미지 경로: `src/images/seagull.jpg`
- 필요한 라이브러리: numpy, matplotlib, opencv-python, torch, torchvision, cupy
- `time`, `random`은 Python 기본 라이브러리입니다.
- MNIST는 최초 실행 시 다운로드하므로, 인터넷 연결이 요구됩니다.
- CuPy와 CUDA를 사용할 수 있는 NVIDIA GPU 환경이 필요합니다.
- 그래프 창이 표시되면 확인 후 닫아야 다음 코드가 실행됩니다.
- 결과는 Matplotlib으로 출력하며, 이미지 파일로 자동 저장하지는 않습니다.

## 4. 아이디어와 구현 내용

### 앞부분 코드 활용

- 성능 그래프는 기존 `plt.plot()`을 사용하고, 정확도 변환은 `np.array()`와 배열 곱셈을 사용했습니다.
- Feature Map은 기존 `show_feature_maps()`를 반복해서 호출하도록 했습니다.
- NumPy 잔상 함수는 앞의 convolution 코드에서 사용한 배열 크기 확인, 출력 배열 생성, 중첩 반복문과 인덱싱 방식을 활용했습니다.
- CuPy 잔상과 convolution은 기본 슬라이싱과 배열 곱셈·누적으로 계산했습니다.
- GPU 배열을 만들 때는 `np.zeros()`에 대응하는 `cp.zeros()`를 사용했습니다.
- 배열 변환과 시간 측정은 실습의 `cp.asarray()`, `cp.asnumpy()`, `time.perf_counter()`와 동기화를 활용했습니다.

### 구현한 부분

- 잔상 계산에서 이미지 밖의 픽셀은 `if input_x >= 0` 조건으로 제외했습니다.
- CuPy 잔상은 이동량이 이미지 너비보다 작은 경우만 계산하고, 결과를 출력 배열의 슬라이스에 직접 대입했습니다.
- GPU convolution은 과제에서 사용하는 `stride=1`에 맞춰 기본 슬라이싱으로 구현했습니다.
- 절대 차이는 `np.maximum(차이, -차이)`로 구하고, `np.max()`로 가장 큰 값을 선택했습니다.
- CPU 처리 시간은 convolution 실행 전후의 시각을 저장한 뒤 차이를 구했습니다.
- GPU는 워밍업 후 계산하고, 동기화를 통해 연산이 끝난 시점까지 시간을 측정했습니다.

## 5. 한계와 참고 사항

- GPU convolution은 `stride=1`에서 사용해야 합니다. 다른 stride 값을 막는 조건은 없으며, 다른 값을 넣으면 잘못된 결과가 나올 수 있습니다.
- Convolution의 최대 절대 차이는 ReLU 적용 후 결과를 비교한 값입니다.
- Convolution 시간에는 두 방향 필터 계산만 포함하고, ReLU, 결과 복사와 시각화는 제외합니다.
- CPU는 픽셀별 반복문을 사용하고 GPU는 배열 전체를 계산하므로, 가속 배율에는 장치 성능뿐 아니라 계산 방식의 차이도 포함됩니다.
- 실제 학습 결과와 GPU 처리 시간은 실행 환경에 따라 달라질 수 있습니다.
- 과제 수행은 Google Colab에서 진행하였습니다. 런타임 유형을 T4 GPU로 변경하여 CUPY 같은 GPU 연산 기능들도 사용할 수 있도록 테스트해보았습니다.
