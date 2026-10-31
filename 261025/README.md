# 261025 과제 - 딥러닝 및 GPU 가속

`03_DL-and-GPU.ipynb`의 모델 성능 확인부터 CuPy convolution 연산까지의 TODO를 구현한 과제입니다.

앞에서 배운 CNN 학습 코드와 Feature Map 시각화 함수를 사용하고, NumPy와 CuPy로 이미지 잔상 효과와 convolution을 구현하여 CPU와 GPU의 결과 및 처리 시간을 비교하는 과제를 수행하였습니다.

<p align="center">
  <img src="pictures/CNN_Accuracy.jpeg" width="24%">
  <img src="pictures/Figure1.jpeg" width="24%">
  <img src="pictures/seagull2.jpeg" width="24%">
  <img src="pictures/Comparison2.jpeg" width="24%">
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

## 2. 동작 과정 및 결과 사진

### 모델 성능 확인

- 기존 `SimpleCNN`, `train_one_epoch()`, `evaluate()`를 사용해 MNIST를 CPU에서 3 Epoch 학습하고 평가합니다.
- 각 Epoch의 Train/Test Loss와 Accuracy를 저장하고, 학습 결과와 소요 시간을 출력합니다.
- 저장한 기록으로 Loss와 Accuracy 그래프를 출력합니다.
- Accuracy는 NumPy 배열로 변환한 뒤 100을 곱해 백분율로 표시합니다.

<p align="center">
  <img src="pictures/CNN_Loss.jpeg" width="24%">
  <img src="pictures/CNN_Accuracy.jpeg" width="24%">
</p>

- Loss 변화 : Epoch가 증가하면서 Train Loss는 0.3133에서 0.0666으로, Test Loss는 0.0953에서 0.0495로 감소했습니다. 학습 데이터뿐 아니라 테스트 데이터에서도 Loss가 감소하는 것을 확인했습니다.
- Accuracy 변화 : Train Accuracy는 90.52%에서 97.92%로, Test Accuracy는 96.99%에서 98.25%로 증가했습니다. 3 Epoch 동안 학습이 진행되면서 분류 정확도가 높아졌으며, CPU에서 전체 학습에 걸린 시간은 약 360.38초였습니다.

### 학습된 모델의 Feature Map 확인

- 테스트 이미지 한 장을 선택하고 Batch 차원을 추가합니다.
- `model.eval()`과 `torch.no_grad()`를 사용해 평가 모드에서 Gradient 계산 없이 순전파합니다.
- `return_features=True`로 각 층의 중간 출력을 가져옵니다.
- 기존 `show_feature_maps()`를 사용해 Conv1은 8개, Conv2는 16개를 출력합니다.
- conv1, relu1, pool1, conv2, relu2, pool2는 각각 4개씩 출력하여 비교합니다.
- 색상은 `viridis`를 사용합니다.

<p align="center">
  <img src="pictures/Figure1.jpeg" width="24%">
  <img src="pictures/Figure2.jpeg" width="24%">
  <img src="pictures/Figure3.jpeg" width="24%">
  <img src="pictures/Figure4.jpeg" width="24%">
</p>

<p align="center">
  <img src="pictures/Figure5.jpeg" width="24%">
  <img src="pictures/Figure6.jpeg" width="24%">
  <img src="pictures/Figure7.jpeg" width="24%">
  <img src="pictures/Figure8.jpeg" width="24%">
</p>

- 첫 번째 convolution 결과에서는 숫자 7의 가로획과 대각선 형태가 나타났습니다. 같은 이미지를 입력해도 채널마다 강조되는 부분과 출력 형태가 다르게 나타나는 것을 확인했습니다.
- 두 번째 convolution 결과에서는 채널에 따라 가로획이나 대각선의 일부가 강조됐습니다. 앞부분에서 추출한 특징이 다음 convolution을 거치면서 서로 다른 형태로 나타났습니다.
- 다음으로는 첫 번째 convolution의 대표 채널 4개를 비교했습니다. 숫자 7의 전체적인 형태가 유지되면서도 채널별로 출력 차이가 나타났습니다.
- ReLU 적용 후 음수 출력이 0으로 바뀌었습니다. 특히 Channel 2는 해당 이미지에 대한 출력 전체가 0으로 처리되어 단색으로 나타났습니다.
- 5번째 사진에서는 Max Pooling으로 Feature Map의 크기가 28×28에서 14×14로 줄었습니다. 세부 표현은 거칠어졌지만, 활성화된 채널에서는 숫자의 주요 형태가 유지됐습니다.
- 두 번째 convolution의 대표 채널 4개를 비교했습니다. 가로획과 대각선 부분에 대한 반응이 채널마다 다르게 나타났습니다.
- ReLU를 적용하면서 음수 출력이 제거되고, 각 채널에서 양수로 반응한 부분이 남았습니다. 채널별로 남아 있는 영역이 다르게 나타났습니다.
- 마지막으로, 두 번째 Max Pooling을 통해 크기가 14×14에서 7×7로 줄었습니다. 세부 모양은 단순해졌지만, 각 채널에서 강하게 반응한 영역은 남는 것을 확인했습니다.

### 이미지 잔상 효과 구현

- 갈매기 이미지를 흑백으로 불러오고 1280×720 크기로 조정합니다.
- 픽셀값을 float32의 0~1 범위로 변환합니다.
- 오른쪽으로 이동할수록 작아지는 가중치 32개를 만들고, 합이 1이 되도록 정규화합니다.
- NumPy에서는 출력 픽셀을 하나씩 계산합니다.
- CuPy에서는 이동량만 반복하고, 각 이동 단계의 픽셀 계산은 배열 전체에 적용합니다.
- CPU 처리 시간, GPU 처리 시간, CPU와 GPU 사이의 복사 시간을 출력합니다.
- 복사 시간을 제외한 가속 배율과 포함한 가속 배율을 각각 구합니다.
- 원본, CPU 잔상, GPU 잔상 이미지를 출력하여 비교합니다.

<p align="center">
  <img src="pictures/seagull1.jpeg" width="33%">
  <img src="pictures/seagull2.jpeg" width="33%">
  <img src="pictures/Comparison1.jpeg" width="33%">
</p>

- 첫 번째 사진은 잔상 효과를 적용하기 전의 흑백 이미지입니다. 갈매기의 눈, 부리와 머리 경계가 비교적 선명하게 나타납니다.
- 두 번째 사진에서는 잔상 효과를 적용한 뒤에 원본보다 경계가 가로 방향으로 퍼지고 흐려졌습니다. CPU와 GPU 결과는 사진상으로 거의 같은 잔상 형태를 보였습니다.
- 결과 터미널에서는 CPU 처리 시간은 약 20.1503초, GPU 처리 시간은 약 0.0316초였으며, 계산 시간 기준으로 약 638.02배의 가속이 기록됐습니다.
- 데이터 복사 시간을 포함한 GPU 전체 시간은 약 0.1900초였고, 이때 가속 배율은 약 106.06배였습니다. 복사 시간을 포함하면 가속 배율이 낮아져, 데이터 이동 시간도 전체 처리 시간에 영향을 준다는 것을 확인했습니다.

### Convolution 연산

- 실습에서 사용한 가로, 세로 edge 필터와 편향을 사용합니다.
- CPU에서는 기존 `convolution2d()`를 사용합니다.
- GPU에서는 커널 원소를 하나씩 선택하고, 대응하는 입력 영역 전체에 가중치를 곱해 누적합니다.
- 0으로 채운 배열에 입력 이미지를 넣어서 zero padding을 구성합니다.
- 두 구현 모두 `stride=1`, `padding=1`로 계산합니다.
- GPU 결과를 `cp.asnumpy()`로 복사한 뒤 기존 `relu()`를 적용합니다.
- CPU/GPU 처리 시간과 가속 배율을 출력합니다.
- ReLU를 적용한 결과끼리 최대 절대 차이를 구하고, 가로·세로 edge 이미지를 비교합니다.

<p align="center">
  <img src="pictures/seagull3.jpeg" width="49%">
  <img src="pictures/Comparison2.jpeg" width="49%">
</p>

- 방향별 필터를 적용한 결과, 갈매기 이미지에서 일부 경계가 검출됐습니다. 결과는 대부분 어둡고 일부 경계만 희미하게 나타났습니다. 음수 편향과 ReLU를 적용하면서 약한 반응과 음수 출력이 제거된 결과로 해석할 수 있습니다.
- CPU 처리 시간은 약 25.2749초, GPU 처리 시간은 약 0.0150초였으며, 출력된 가속 배율은 약 1682.78배였습니다.
- ReLU 적용 후 최대 절대 차이는 가로, 세로 방향 모두 0.000000으로 출력됐습니다. 출력된 소수점 여섯 자리 기준으로 CPU와 GPU 결과의 차이가 나타나지 않았음을 볼 수 있습니다.

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
- CuPy 잔상과 convolution은 기본 슬라이싱과 배열 곱셈, 누적으로 계산했습니다.
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
- 최종 테스트는 실습실의 엔비디아 젯슨 오린 나노에서 직접 진행해보았습니다. 진행한 결과를 찍어서 기록으로 남겼습니다.
