### 261025_task.py 독립 실행을 위해, 실습 파일에서 필요한 내용을 가져왔습니다.

import time
import random

import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import cupy as cp


def relu(x):
    return np.maximum(0, x)

def convolution2d(image, kernel, bias=0.0, stride=1, padding=0):
    """
    2차원 이미지에 하나의 CNN 필터를 적용.

    Parameters
    ----------
    image : np.ndarray
        입력 이미지, shape = (height, width)
    kernel : np.ndarray
        CNN 필터, shape = (kernel_height, kernel_width)
    bias : float
        필터 출력에 더할 편향
    stride : int
        필터가 이동하는 간격
    padding : int
        입력 이미지 가장자리에 추가할 픽셀의 개수

    Returns
    -------
    output : np.ndarray
        편향까지 적용된 convolution 결과
    """

    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # 이미지 가장자리에 zero padding 적용
    padded_image = np.pad(
        image,
        ((padding, padding), (padding, padding)),
        mode="constant",
        constant_values=0
    )

    # (H, W: 입력 이미지 높이 및 너비, P: Padding 픽셀 값, S: Stride 픽셀 값)
    # Convolution 출력 이미지 높이 = floor((H + 2P - K) / S) + 1
    output_height = (padded_image.shape[0] - kernel_height) // stride + 1

    # Convolution 출력 이미지 너비 = floor((W + 2P - K) / S) + 1
    output_width = (padded_image.shape[1] - kernel_width) // stride + 1

    # 계산한 크기로 출력 이미지 준비
    output = np.zeros((output_height, output_width), dtype=np.float32)

    # 커널을 움직이며 필터 적용
    for output_y in range(output_height):
        for output_x in range(output_width):
            # 커널을 적용할 이미지 영역의 좌측상단
            start_y = output_y * stride
            start_x = output_x * stride

            # 커널을 적용할 이미지 영역을 crop
            image_region = padded_image[
                start_y:start_y + kernel_height,
                start_x:start_x + kernel_width
            ]

            # 필터를 적용하여 가중합 계산
            weighted_sum = np.sum(image_region * kernel)

            # 편향 적용
            output[output_y, output_x] = weighted_sum + bias

    return output

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        self.fc1 = nn.Linear(
            16 * 7 * 7,
            64,
        )

        self.fc2 = nn.Linear(
            64,
            10,
        )

    def forward(self, x, return_features=False):
        conv1_output = self.conv1(x)
        relu1_output = F.relu(conv1_output)
        pool1_output = self.pool(relu1_output)
        conv2_output = self.conv2(pool1_output)
        relu2_output = F.relu(conv2_output)
        pool2_output = self.pool(relu2_output)
        flattened = torch.flatten(pool2_output, start_dim=1)
        fc1_output = F.relu(self.fc1(flattened))
        logits = self.fc2(fc1_output)

        if return_features:
            features = {
                "conv1": conv1_output,
                "relu1": relu1_output,
                "pool1": pool1_output,
                "conv2": conv2_output,
                "relu2": relu2_output,
                "pool2": pool2_output,
            }

            return logits, features

        return logits

def show_feature_maps(feature_tensor, title, max_maps=8, cmap="gray"):
    if feature_tensor.ndim != 4:
        raise ValueError("Feature Tensor는 [Batch, Channel, Height, Width] 형태여야 합니다.")

    feature_maps = feature_tensor[0].detach().cpu()

    channel_count = min(feature_maps.shape[0], max_maps)
    column_count = 4
    row_count = int(np.ceil(channel_count / column_count))

    plt.figure(figsize=(12, 3 * row_count))

    for channel_index in range(channel_count):
        plt.subplot(row_count, column_count, channel_index + 1)
        plt.imshow(feature_maps[channel_index], cmap=cmap)
        plt.title(f"Channel {channel_index}")
        plt.axis("off")
        plt.suptitle(title)

def train_one_epoch(model, data_loader, criterion, optimizer, device):
    # 모델을 학습 모드(training mode)로 변경
    model.train()

    # 결과를 누적하기 위한 변수
    total_loss = 0.0
    correct_count = 0
    sample_count = 0

    # 학습 데이터를 DataLoader(train_loader)의 Batch 단위로 불러와 학습 반복
    for images, labels in data_loader:
        # CPU로 이동
        images = images.to(device)
        labels = labels.to(device)

        # Gradient 초기화 (이전 Batch에서 계산한 Gradient를 제거)
        optimizer.zero_grad()

        # 순전파를 통해 원시 출력 저장 (클래스별 점수 10개)
        logits = model(images)

        # Loss 계산
        loss = criterion(logits, labels)
        # 역전파를 통한 Gradient 계산
        loss.backward()
        # Gradient를 사용하여 파라미터 수정
        optimizer.step()

        # Batch 한 개의 이미지 개수
        batch_size = images.size(0)

        # 결과 누적
        total_loss += loss.item() * batch_size                 # Loss 누적 (1개 batch의 평균 loss x batch 크기)
        predictions = logits.argmax(dim=1)                     # 가장 높은 점수를 가진 클래스를 선택
        correct_count += (predictions == labels).sum().item()  # 정답을 맞춘 개수 계산
        sample_count += batch_size                             # 전체 샘플 수 누적

    # 평균 Loss 및 Accuracy 계산
    average_loss = total_loss / sample_count
    accuracy = correct_count / sample_count

    return average_loss, accuracy

def evaluate(model, data_loader, criterion, device):
    # 모델을 평가 모드(evaluation mode)로 변경
    model.eval()

    # 결과를 누적하기 위한 변수
    total_loss = 0.0
    correct_count = 0
    sample_count = 0

    # Gradient 계산을 비활성화 (파라미터 수정 비활성화)
    with torch.no_grad():
        # 평가 데이터를 DataLoader(test_loader)의 Batch 단위로 불러와 평가 반복
        for images, labels in data_loader:
            # CPU로 이동
            images = images.to(device)
            labels = labels.to(device)

            # 학습된 모델의 순전파를 통해 원시 출력 저장 (클래스별 점수 10개)
            logits = model(images)

            # Loss 계산 (역전파는 수행하지 않음)
            loss = criterion(logits, labels)

            # Batch 한 개의 이미지 개수
            batch_size = images.size(0)

            # 결과 누적
            total_loss += loss.item() * batch_size                 # Loss 누적 (1개 batch의 평균 loss x batch 크기)
            predictions = logits.argmax(dim=1)                     # 가장 높은 점수를 가진 클래스를 선택
            correct_count += (predictions == labels).sum().item()  # 정답을 맞춘 개수 계산
            sample_count += batch_size                             # 전체 샘플 수 누적

        # 평균 Loss 및 Accuracy 계산
        average_loss = total_loss / sample_count
        accuracy = correct_count / sample_count

        return average_loss, accuracy


seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed);

# 데이터를 불러올 때 이미지 전처리 방식 정의
# PIL 이미지 또는 NumPy 이미지를 PyTorch Tensor로 변환
# 일반적인 8비트 이미지의 픽셀값을 0~1 범위의 실수로 변환
transform = transforms.ToTensor()

# 학습 데이터 불러오기 (최초 다운로드)
train_dataset = datasets.MNIST(
    root="src/datasets",
    train=True,
    transform=transform,
    download=True,
)

# 테스트 데이터 불러오기 (최초 다운로드)
test_dataset = datasets.MNIST(
    root="src/datasets",
    train=False,
    transform=transform,
    download=True,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1000,
    shuffle=False,
    num_workers=0,
)

device = torch.device("cpu")

torch.manual_seed(seed)

model = SimpleCNN()
model = model.to(device)

print("Device:", device)
print(model)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001, #learnig rate, 학습률
)

# 총 Epoch 정의 (전체 학습 데이터를 몇 번 반복해서 볼 것인지)
epochs = 3

# 학습 기록 저장 공간 준비
train_loss_history = []
train_accuracy_history = []
test_loss_history = []
test_accuracy_history = []

# 전체 학습 시간 측정 시작
training_start_time = time.perf_counter()

for epoch in range(epochs):
    # Epoch 시작 시간 저장
    epoch_start_time = time.perf_counter()

    # 모델 학습
    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device,
    )

    # 모델 평가
    test_loss, test_accuracy = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )

    # 결과 저장 (학습 & 평가)
    train_loss_history.append(train_loss)
    train_accuracy_history.append(train_accuracy)
    test_loss_history.append(test_loss)
    test_accuracy_history.append(test_accuracy)

    # Epoch 소요 시간 계산
    epoch_time = time.perf_counter() - epoch_start_time

    # 현재 Epoch의 학습/평가 결과 출력
    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Train loss: {train_loss:.4f} | "
        f"Train accuracy: "
        f"{train_accuracy * 100:.2f}% | "
        f"Test loss: {test_loss:.4f} | "
        f"Test accuracy: "
        f"{test_accuracy * 100:.2f}% | "
        f"Time: {epoch_time:.2f}s"
    )

# 전체 학습 시간 계산 및 출력
total_training_time = time.perf_counter() - training_start_time
print(f"Total CPU training time: {total_training_time:.2f}s")

### 여기서부터 과제!!

### 모델 성능 확인

# TODO: plt.plot(x축_데이터, y축_데이터, marker="o", label="데이터_이름")을 사용하여 그래프 생성
plt.figure(figsize=(8,5))

# TODO: train_loss_history 그래프 생성
plt.plot(
    range(1, epochs + 1), 
    train_loss_history, 
    marker="o", 
    label="Train Loss"
)
# TODO: test_loss_history 그래프 생성
plt.plot(
    range(1, epochs + 1), 
    test_loss_history, 
    marker="o", 
    label="Test Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CNN Loss")
plt.grid()
plt.legend()
plt.show()

# TODO: plt.plot(x축_데이터, y축_데이터, marker="o", label="데이터_이름")을 사용하여 그래프 생성
plt.figure(figsize=(8,5))

# TODO: train_accuracy_history 그래프 생성 (백분율)
plt.plot(
    range(1, epochs + 1),
    np.array(train_accuracy_history) * 100,
    marker="o",
    label="train_accuracy_history",
)

# TODO: test_accuracy_history 그래프 생성 (백분율)
plt.plot(
    range(1, epochs + 1),
    np.array(test_accuracy_history) * 100,
    marker="o",
    label="test_accuracy_history",
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("CNN Accuracy")
plt.grid()
plt.legend()
plt.show()


### 학습된 모델의 Feature Map 확인

# TODO: CNN 각 층의 Feature Map 시각화
# TODO: cmap="viridis" 사용


# TODO: 이미지 선택
image, true_label = test_dataset[0]
input_batch = image.unsqueeze(0).to(device)

# TODO: 역전파(파라미터 수정) 없이 순전파 수행
model.eval()
with torch.no_grad():
    logits, features = model(
        input_batch,
        return_features=True,
    )

# TODO: Conv1 Feature Maps 8개 시각화
show_feature_maps(
    features["conv1"],
    title="Trained Conv1 Feature Maps",
    max_maps=8,
    cmap="viridis",
)

# TODO: Conv2 Feature Maps 16개 시각화
show_feature_maps(
    features["conv2"],
    title="Trained Conv2 Feature Maps",
    max_maps=16,
    cmap="viridis",
)

# TODO: 6개 Layer별 대표 Feature Map 4개씩 비교
layer_names = [
    "conv1",
    "relu1",
    "pool1",
    "conv2",
    "relu2",
    "pool2"
]

for layer_name in layer_names:
    show_feature_maps(
        features[layer_name],
        title=layer_name,
        max_maps=4,
        cmap="viridis",
    )


plt.show()

### 이미지 잔상 효과 구현

IMAGE_PATH = "src/images/seagull.jpg"

TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

TRAIL_LENGTH = 32

img_gray = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

if img_gray is None:
    raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {IMAGE_PATH}")

img_gray = cv2.resize(
    img_gray,
    (TARGET_WIDTH, TARGET_HEIGHT),
    interpolation=cv2.INTER_AREA
)

input_np = img_gray.astype(np.float32) / 255.0

plt.imshow(input_np, cmap="gray")
plt.axis("off");

# TODO: NumPy를 사용하여 잔상 효과 함수 정의


def motion_trail_np(image, weights):
    """
    NumPy를 이용해 이미지 잔상 효과를 생성한다.

    출력 픽셀을 Python 반복문으로 하나씩 계산한다.
    """


    # TODO: 필요한 변수 정의 및 출력 배열 생성
    image_height, image_width = image.shape
    output = np.zeros((image_height, image_width), dtype=np.float32)

    for output_y in range(image_height):
        for output_x in range(image_width):

            pixel_sum = 0.0


            for shift in range(len(weights)):
                # TODO: x값 shift

                input_x = output_x - shift


                if input_x >= 0:
                    # TODO: shift한 픽셀을 누적
                    pixel_sum += image[output_y, input_x] * weights[shift]

            # TODO: 출력 배열에 누적 픽셀 저장
            output[output_y, output_x] = pixel_sum

    return output

# TODO: CuPy를 사용하여 잔상 효과 함수 정의


def motion_trail_cp(image, weights):
    """
    CuPy를 이용해 이미지 잔상 효과를 생성한다.

    이동 거리는 Python에서 순차적으로 반복하지만,
    각 이동 거리의 이미지 전체 계산은 GPU에서 병렬 처리한다.
    """

    # TODO: 필요한 변수 정의 및 출력 배열 생성
    image_height, image_width = image.shape
    output = cp.zeros((image_height, image_width), dtype=cp.float32)

    for shift in range(len(weights)):

        if shift < image_width:

            # TODO: 배열 slicing
            input_region = image[:, 0:image_width - shift]


            # TODO: GPU가 병렬 연산할 output_region 계산 코드
            output[:, shift:image_width] = (
                output[:, shift:image_width] + input_region * weights[shift]
            )

    return output

# 이동 거리가 멀어질수록 가중치를 작게 설정
weights_np = np.linspace(
    1.0,
    0.05,
    TRAIL_LENGTH,
    dtype=np.float32
)

# 전체 가중치의 합을 1로 정규화
weights_np /= np.sum(weights_np)

cpu_start = time.perf_counter()

output_np = motion_trail_np(
    image=input_np,
    weights=weights_np
)

cpu_end = time.perf_counter()

cpu_time = cpu_end - cpu_start

# TODO: NumPy 배열을 CuPy 배열로 변환

upload_start = time.perf_counter()

input_cp = cp.asarray(input_np)
weights_cp = cp.asarray(weights_np)

cp.cuda.Stream.null.synchronize()

upload_end = time.perf_counter()

upload_time = upload_end - upload_start

warmup_image = cp.zeros((64, 64), dtype=cp.float32)
warmup_weights = cp.asarray(weights_np[:4])

_ = motion_trail_cp(
    image=warmup_image,
    weights=warmup_weights
)

cp.cuda.Stream.null.synchronize()

cp.cuda.Stream.null.synchronize()

gpu_start = time.perf_counter()

output_cp = motion_trail_cp(
    image=input_cp,
    weights=weights_cp
)

cp.cuda.Stream.null.synchronize()

gpu_end = time.perf_counter()

gpu_time = gpu_end - gpu_start

# TODO: output_cp를 CPU 메모리로 복사

download_start = time.perf_counter()

output_cp_np = cp.asnumpy(output_cp)

download_end = time.perf_counter()

download_time = download_end - download_start

compute_speedup = cpu_time / gpu_time

gpu_total_time = upload_time + gpu_time + download_time
total_speedup = cpu_time / gpu_total_time

print("========== NumPy CPU ==========")
print(f"Processing time: {cpu_time:.4f} seconds")
print()

print("========== CuPy GPU ==========")
print(f"CPU to GPU: {upload_time:.4f} seconds")
print(f"GPU processing: {gpu_time:.4f} seconds")
print(f"GPU to CPU: {download_time:.4f} seconds")
print(f"Total GPU time: {gpu_total_time:.4f} seconds")
print()

print("========== Comparison ==========")
print(f"Compute-only speedup: {compute_speedup:.2f}x")
print(f"Including transfers: {total_speedup:.2f}x")

plt.figure(figsize=(18,12))

plt.subplot(1,3,1), plt.imshow(input_np, cmap="gray"), plt.title("Original Image")
plt.subplot(1,3,2), plt.imshow(output_np, cmap="gray"), plt.title("Motion Trail (CPU)")
plt.subplot(1,3,3), plt.imshow(output_cp_np, cmap="gray"), plt.title("Motion Trail (GPU)")

for ax in plt.gcf().axes:
    ax.axis("off")

plt.show()

### Convolution 연산

# x축 방향 edge (세로 선) 검출 필터
dx_edge_filter = np.array(
    [
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ],
    dtype=np.float32
)

# y축 방향 edge (가로 선) 검출 필터
dy_edge_filter = np.array(
    [
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]
    ],
    dtype=np.float32
)

# 너무 약한 특징 (0.2 이하)은 무시
dx_bias = -0.2
dy_bias = -0.2

# TODO: CuPy를 사용하여 2차원 convolution 함수 정의
# 힌트 : 픽셀별로 움직이는 것이 아니다!! 1 2번 다!

def convolution2d_cp(image, kernel, bias=0.0, stride=1, padding=0):
    """
    CuPy를 이용한 2차원 convolution.

    커널의 위치는 Python 반복문으로 순차 처리하고,
    각 위치에 해당하는 출력 배열 전체 연산은 GPU에서 처리한다.
    """

    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape


    padded_image = cp.zeros(
        (image_height + 2 * padding, image_width + 2 * padding),
        dtype=cp.float32,
    )
    padded_image[
        padding:padding + image_height,
        padding:padding + image_width
    ] = image


    output_height = (padded_image.shape[0] - kernel_height) // stride + 1
    output_width = (padded_image.shape[1] - kernel_width) // stride + 1
    output = cp.zeros((output_height, output_width), dtype=cp.float32)


    for kernel_y in range(kernel_height):
        for kernel_x in range(kernel_width):

            input_region = padded_image[
                kernel_y:kernel_y + output_height,
                kernel_x:kernel_x + output_width
            ]
            output += input_region * kernel[kernel_y, kernel_x]

    output += bias
    return output

# TODO: 기존에 정의한 convolution2d 함수를 사용하여 convolution을 CPU로 처리
# TODO: 변수 재사용 가능

cpu_start = time.perf_counter()

output_dx_np = convolution2d(
    image=input_np,
    kernel=dx_edge_filter,
    bias=dx_bias,
    stride=1,
    padding=1
)

output_dy_np = convolution2d(
    image=input_np,
    kernel=dy_edge_filter,
    bias=dy_bias,
    stride=1,
    padding=1
)

cpu_end = time.perf_counter()

cpu_time = cpu_end - cpu_start
relu_dx_np = relu(output_dx_np)
relu_dy_np = relu(output_dy_np)

# TODO: convolution을 GPU로 처리 (변환-워밍업-처리 파이프라인 구현)
# TODO: 변수 재사용 가능


input_cp = cp.asarray(input_np)
dx_edge_filter_cp = cp.asarray(dx_edge_filter)
dy_edge_filter_cp = cp.asarray(dy_edge_filter)

cp.cuda.Stream.null.synchronize()


warmup_image = cp.zeros((64, 64), dtype=cp.float32)

_ = convolution2d_cp(
    image=warmup_image,
    kernel=dx_edge_filter_cp,
    bias=dx_bias,
    stride=1,
    padding=1
)

cp.cuda.Stream.null.synchronize()


gpu_start = time.perf_counter()

output_dx_cp = convolution2d_cp(
    image=input_cp,
    kernel=dx_edge_filter_cp,
    bias=dx_bias,
    stride=1,
    padding=1
)
output_dy_cp = convolution2d_cp(
    image=input_cp,
    kernel=dy_edge_filter_cp,
    bias=dy_bias,
    stride=1,
    padding=1
)

cp.cuda.Stream.null.synchronize()

gpu_end = time.perf_counter()

gpu_time = gpu_end - gpu_start


output_dx_cp_np = cp.asnumpy(output_dx_cp)
output_dy_cp_np = cp.asnumpy(output_dy_cp)


relu_dx_cp_np = relu(output_dx_cp_np)
relu_dy_cp_np = relu(output_dy_cp_np)

# TODO: CPU/GPU 결과 비교 및 출력

speedup = cpu_time / gpu_time


dx_delta = relu_dx_np - relu_dx_cp_np
dy_delta = relu_dy_np - relu_dy_cp_np
dx_difference = np.max(np.maximum(dx_delta, -dx_delta))
dy_difference = np.max(np.maximum(dy_delta, -dy_delta))


print("========== NumPy CPU ==========")
print(f"Processing time: {cpu_time:.4f} seconds")
print()

print("========== CuPy GPU ==========")
print(f"Processing time: {gpu_time:.4f} seconds")
print()

print("========== Comparison ==========")
print(f"Speedup: {speedup:.2f}x")
print(f"Vertical (dx) max difference: {dx_difference:.6f}")
print(f"Horizontal (dy) max difference: {dy_difference:.6f}")

plt.figure(figsize=(18,8))

plt.subplot(2,3,1), plt.imshow(input_np, cmap="gray"), plt.title("Input Image")
plt.subplot(2,3,2), plt.imshow(relu_dx_np, cmap="gray"), plt.title("Vertical Convolution (CPU)")
plt.subplot(2,3,3), plt.imshow(relu_dx_cp_np, cmap="gray"), plt.title("Vertical Convolution (GPU)")
plt.subplot(2,3,4), plt.imshow(input_np, cmap="gray"), plt.title("Input Image")
plt.subplot(2,3,5), plt.imshow(relu_dy_np, cmap="gray"), plt.title("Horizontal Convolution (CPU)")
plt.subplot(2,3,6), plt.imshow(relu_dy_cp_np, cmap="gray"), plt.title("Horizontal Convolution (GPU)")

for ax in plt.gcf().axes:
    ax.axis("off")

plt.show()


