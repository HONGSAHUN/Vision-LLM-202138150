# 261025 과제 - 딥러닝 GPU 가속과 객체 탐지 기초

`03_DL-and-GPU.ipynb`와 `04_DL-Object-Detection.ipynb`의 과제를 구현했습니다. 지난주 과제(CuPy 잔상 효과, Convolution)와 이번 주 과제(MNIST·CIFAR-10 GPU 학습, IoU, NMS)를 함께 제출합니다.

| 과제 | 파일 |
|---|---|
| MNIST 성능 확인, Feature Map, 모델 저장 | `261025_task.py` |
| GPU 잔상 효과, GPU Convolution (CuPy) | `261025_task.py` |
| MNIST `main_gpu()`, CIFAR-10 학습 및 모델 저장 | `261025_gpu_training.py` |
| IoU, Confidence Thresholding, NMS | `261025_iou_nms.py` |

## 1. 파일 구조

```text
261025/
├── 261025_task.py             # 03: 모델 성능 확인 ~ CuPy Convolution
├── 261025_gpu_training.py     # 03: MNIST main_gpu(), CIFAR-10 학습
├── 261025_iou_nms.py          # 04: IoU, NMS
├── README.md
├── pictures/                  # 결과 사진
└── src/
    ├── images/                # seagull.jpg, two_apples1~3.png
    └── models/
        ├── MNIST/MNIST_CNN.pth        # 학습한 MNIST 모델 (가중치)
        └── CIFAR10/CIFAR10_CNN.pth    # 학습한 CIFAR-10 모델 (가중치 + 학습 정보)
```

- 각 파일은 실습 파일에서 필요한 함수와 모델 정의를 가져와, 261025 폴더만으로 따로 실행할 수 있습니다.
- MNIST, CIFAR-10 데이터는 최초 실행 시 `src/datasets`에 다운로드되며, 용량이 커서 제출에는 포함하지 않았습니다.

## 2. 실행 방법

261025 폴더 안에서 실행합니다.

```bash
python 261025_task.py
python 261025_gpu_training.py
python 261025_iou_nms.py
```

- 필요한 라이브러리: numpy, matplotlib, opencv-python, torch, torchvision, cupy, scikit-image
- `261025_task.py`, `261025_gpu_training.py`는 CUDA GPU가 필요하고, `261025_iou_nms.py`는 GPU 없이 실행됩니다.
- 데이터셋 다운로드에 인터넷 연결이 필요하며, CIFAR-10(약 163MB)은 다운로드가 오래 걸릴 수 있습니다.
- 그래프 창을 닫아야 다음 코드가 실행됩니다.

**제출한 모델 불러오기** (모델 클래스는 실습 파일과 같은 정의 사용)

```python
import torch

mnist_model = SimpleCNN()
mnist_model.load_state_dict(
    torch.load("src/models/MNIST/MNIST_CNN.pth", map_location="cpu", weights_only=True)
)

checkpoint = torch.load("src/models/CIFAR10/CIFAR10_CNN.pth", map_location="cpu", weights_only=True)
cifar_model = CIFAR10CNN()
cifar_model.load_state_dict(checkpoint["model_state_dict"])
print(checkpoint["epochs"], checkpoint["test_accuracy"])   # 5, 0.8082
```

## 3. 구현 내용과 결과

### 3-1. MNIST 성능 확인과 Feature Map

- `SimpleCNN`을 CPU에서 3 Epoch 학습하고, Epoch별 Loss와 Accuracy를 그래프로 출력합니다.
- 학습한 모델에 테스트 이미지 한 장을 넣어 `return_features=True`로 각 층의 출력을 가져오고, `show_feature_maps()`로 시각화합니다(`viridis`).
- 학습한 가중치를 `src/models/MNIST/MNIST_CNN.pth`에 저장하고, 다시 불러와 확인합니다.

<p align="center">
  <img src="pictures/CNN_Loss.png" width="32%">
  <img src="pictures/CNN_Accuracy.png" width="32%">
</p>

- Train Loss는 0.3133 → 0.0666, Test Loss는 0.0953 → 0.0495로 감소했습니다.
- Train Accuracy는 90.52% → 97.92%, Test Accuracy는 96.99% → 98.25%로 증가했고, CPU 학습 시간은 약 360.38초였습니다.

<p align="center">
  <img src="pictures/Figure1.png" width="24%">
  <img src="pictures/Figure2.png" width="24%">
  <img src="pictures/Figure3.png" width="24%">
  <img src="pictures/Figure4.png" width="24%">
</p>
<p align="center">
  <img src="pictures/Figure5.png" width="24%">
  <img src="pictures/Figure6.png" width="24%">
  <img src="pictures/Figure7.png" width="24%">
  <img src="pictures/Figure8.png" width="24%">
</p>

- Convolution 결과는 같은 숫자 7을 입력해도 채널마다 가로획, 대각선 등 강조되는 부분이 달랐습니다.
- ReLU 후에는 음수 출력이 0이 되어, Conv1의 Channel 2처럼 출력 전체가 0인 채널도 나타났습니다.
- Max Pooling을 거치며 크기가 28×28 → 14×14 → 7×7로 줄었고, 세부 모양은 단순해졌지만 강하게 반응한 영역은 유지됐습니다.

### 3-2. GPU 가속 방식 (잔상 효과, Convolution)

| | CPU (NumPy) | GPU (CuPy) |
|---|---|---|
| 잔상 효과 | 출력 픽셀마다 이동 거리 32개를 반복 | 이동 거리 32번만 반복 |
| Convolution | 출력 픽셀마다 3×3 영역의 가중합 계산 | 커널 원소 9번만 반복 |

- CPU는 1280×720 이미지의 출력 픽셀 약 92만 개를 Python 반복문으로 하나씩 계산합니다.
- GPU는 반복할 때마다 이미지 전체를 슬라이싱한 배열에 가중치를 곱해 출력 배열에 더합니다. 이 배열 연산을 GPU가 픽셀 단위로 병렬 처리하므로 빨라집니다.
- 정확한 측정을 위해 작은 배열로 워밍업을 한 뒤, 측정 전후에 `cp.cuda.Stream.null.synchronize()`로 GPU 연산이 끝날 때까지 기다렸습니다.

### 3-3. 이미지 잔상 효과

- 갈매기 이미지를 흑백, 1280×720, float32(0~1)로 준비하고, 멀어질수록 작아지는 가중치 32개를 합이 1이 되도록 정규화합니다.
- NumPy는 이미지 밖 픽셀을 `if input_x >= 0`으로 제외하고, CuPy는 이동량만큼 어긋난 슬라이스끼리 더합니다.

<p align="center">
  <img src="pictures/seagull1.png" width="33%">
  <img src="pictures/seagull2.png" width="33%">
  <img src="pictures/Comparison1.png" width="33%">
</p>

| CPU 처리 | GPU 처리 | 가속 (계산만) | GPU 전체 (복사 포함) | 가속 (복사 포함) |
|---|---|---|---|---|
| 20.1503초 | 0.0316초 | 638.02배 | 0.1900초 | 106.06배 |

- 경계가 가로 방향으로 퍼지는 잔상이 생겼고, CPU와 GPU 결과는 같은 모양이었습니다.
- 복사 시간을 포함하면 가속 배율이 크게 낮아져, CPU와 GPU 사이 데이터 이동도 전체 시간에 영향을 준다는 것을 확인했습니다.

### 3-4. Convolution

- 실습의 가로·세로 edge 필터와 편향(-0.2)을 `stride=1`, `padding=1`로 적용합니다.
- CPU는 기존 `convolution2d()`를, GPU는 0으로 채운 배열에 이미지를 넣어 zero padding을 만든 뒤 커널 원소별로 누적합니다.
- 두 결과에 `relu()`를 적용한 뒤 최대 절대 차이로 CPU와 GPU 결과를 비교합니다.

<p align="center">
  <img src="pictures/seagull3.png" width="49%">
  <img src="pictures/Comparison2.png" width="49%">
</p>

| CPU 처리 | GPU 처리 | 가속 배율 | 최대 절대 차이 (가로, 세로) |
|---|---|---|---|
| 25.2749초 | 0.0150초 | 1682.78배 | 0.000000, 0.000000 |

- 음수 편향과 ReLU로 약한 반응이 제거되어, 갈매기의 일부 경계만 검출됐습니다.
- 소수점 여섯 자리 기준으로 CPU와 GPU 결과가 같았습니다.

### 3-5. MNIST GPU 학습 (`main_gpu()`)

- `main_cpu()`와 같은 데이터, 모델, Optimizer를 쓰고 장치만 `cuda`로 바꿨습니다. 모델은 `model.to(device)`, 배치는 `train_one_epoch()`, `evaluate()` 안에서 GPU로 옮겨집니다.
- GPU 연산은 비동기이므로 시간 측정 전후에 `torch.cuda.synchronize()`를 호출하고, 전체 학습 시간을 반환해 CPU와 비교합니다.

| 실행 환경 (3 Epoch) | CPU 학습 | GPU 학습 | 가속 배율 |
|---|---|---|---|
| Google Colab (T4 GPU) | 56.29초 | 30.45초 | 1.85배 |
| Jetson Orin Nano (실습 노트북) | 514.61초 | 88.59초 | 5.81배 |

- 같은 시드를 사용해 CPU와 GPU의 정확도는 거의 같았습니다(Colab, 3 Epoch 후 Train Accuracy CPU 97.89%, GPU 97.93%).
- Colab은 CPU가 빨라 가속 배율이 낮게 나왔고, MNIST처럼 작은 모델은 GPU 이점이 크지 않을 수 있습니다.

### 3-6. CIFAR-10 GPU 학습과 모델 저장

- 학습 데이터에만 데이터 증강(`RandomCrop`, `RandomHorizontalFlip`)을, 학습·테스트 모두에 정규화를 적용합니다.
- `ConvBlock` 3개, Global Average Pooling, Dropout, Linear로 구성된 `CIFAR10CNN`을 GPU에서 Warm-up 후 학습합니다.
- 학습 설정: 5 Epoch, Batch 32(학습)/128(테스트), `AdamW`(lr 0.001), `CrossEntropyLoss`, Dropout 0.3, 시드 42, 파라미터 1,148,874개
- 학습 후 모델 가중치, Optimizer 상태, 테스트 정확도, Epoch 수, 클래스 이름, 정규화 값을 `CIFAR10_CNN.pth`에 저장하고, 다시 불러와 테스트 이미지 40장을 분류합니다.

<p align="center">
  <img src="pictures/CIFAR_Loss.png" width="32%">
  <img src="pictures/CIFAR_Accuracy.png" width="32%">
  <img src="pictures/CIFAR_Prediction.png" width="32%">
</p>

- Google Colab(T4 GPU)에서 학습했으며, 저장된 최종 Test Accuracy는 **80.82%** 입니다.
- 그래프 기준으로 Train Loss는 약 1.29 → 0.56, Test Loss는 약 0.94 → 0.58로 감소했고, 마지막 Epoch에서 Train과 Test 정확도가 비슷해 과적합은 크지 않았습니다.
- 초반에 Test Accuracy가 더 높은 것은 학습할 때만 데이터 증강과 Dropout이 적용되기 때문으로 해석할 수 있습니다.
- 불러온 모델로 40장 중 32장을 맞혔고, 개↔말·사슴·개구리, 자동차↔트럭처럼 비슷한 클래스에서 틀렸습니다.

### 3-7. IoU

- `convert_bbox_coord()`: `(x, y, w, h)` → `(x1, y1, x2, y2) = (x, y, x + w, y + h)`
- `calculate_iou()`: 교집합은 좌표의 `max`/`min`으로 구하고, 겹치지 않으면 `max(0, ...)`으로 넓이를 0으로 만듭니다. 합집합은 `두 넓이의 합 - 교집합`이며, 합집합이 0이면 0을 반환해 `ZeroDivisionError`를 막습니다.

<p align="center">
  <img src="pictures/IoU.png" width="90%">
</p>

- 겹치지 않는 경우 0.0000, 나머지 두 경우 0.3046, 0.4934로 실습 파일의 안내값과 일치했습니다.

### 3-8. Confidence Thresholding과 NMS

- `prune_low_conf()`: Confidence 0.5 이상인 BBox만 남깁니다.
- `apply_nms()`: `np.argsort()`로 Confidence가 높은 순으로 정렬한 뒤, 가장 높은 BBox를 선택하고 `calculate_iou()`로 IoU가 임계값(0.2)보다 큰 BBox를 제거하는 과정을 반복합니다.
- 결과를 OpenCV `cv2.dnn.NMSBoxes()`(`apply_nms_cv2()`)와 비교합니다.

<p align="center">
  <img src="pictures/NMS_before.png" width="40%">
  <img src="pictures/NMS_comparison.png" width="58%">
</p>

- BBox 20개 → Thresholding 후 17개 → NMS 후 4개(Confidence 0.80, 0.72, 0.64, 0.62)가 남았습니다.
- 직접 구현한 NMS와 OpenCV NMS의 결과가 같았습니다(`Custom NMS == CV2 NMS: True`).

**실습 코드에서 바꾼 점**
- 실습 코드는 Thresholding 결과 대신 원래 `boxes`에 NMS를 적용하고 있어, Thresholding 결과에 적용하도록 바꿨습니다.
- 결과 그림에 원래 `scores`가 아닌, 남은 BBox의 점수가 표시되도록 바꿨습니다.
- 랜덤 BBox 결과를 다시 확인할 수 있도록 `random.seed(42)`로 난수를 고정했습니다.

## 4. 한계와 참고 사항

- 학습 시간과 가속 배율은 실행 환경에 따라 크게 달라집니다. 개발은 Google Colab(T4 GPU)에서, 잔상 효과와 Convolution의 최종 테스트는 실습실 Jetson Orin Nano에서 진행했습니다.
- CPU는 픽셀별 반복문, GPU는 배열 연산을 사용하므로 가속 배율에는 장치 성능뿐 아니라 계산 방식의 차이도 포함됩니다.
- Convolution 시간에는 두 방향 필터 계산만 포함하고, ReLU, 결과 복사, 시각화는 제외했습니다.
- GPU Convolution은 `stride=1` 기준으로 구현했으며, 다른 stride 값은 막지 않아 잘못된 결과가 나올 수 있습니다.
- CIFAR-10은 5 Epoch만 학습해 충분히 수렴하지 않았습니다.
- NMS에서 Confidence가 완전히 같은 BBox가 있으면 OpenCV와 정렬 순서가 달라 결과가 다를 수 있습니다.
- IoU는 `x2 - x1`로 너비를 계산하며, 경계 픽셀을 1칸 더 세는 방식(`x2 - x1 + 1`)은 사용하지 않았습니다.
