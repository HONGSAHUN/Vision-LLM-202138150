"""
HusAero-8n → TensorRT INT8 엔진 변환 (Jetson에서 실행)
04_DL-Object-Detection.ipynb의 INT8 변환 방식(quantize=8)을 HusAero-8n에 그대로 적용

실행 (레포 최상위 폴더에서, 노트북 커널은 모두 끄고)
    python Project02/husaero_int8.py

결과
    src/models/YOLO/husaero/HusAero-8n_int8.engine
"""

from pathlib import Path
import yaml

from ultralytics import YOLO


### 경로 설정 (project02_261108.py와 같은 방식 : 이 파일의 상위 폴더 = 레포 최상위)
ROOT = Path(__file__).resolve().parent.parent

PT_PATH = ROOT / "src/models/YOLO/husaero/HusAero-8n.pt"
ENGINE_PATH = ROOT / "src/models/YOLO/husaero/HusAero-8n_int8.engine"

### 캘리브레이션 데이터 선택
# "husaero" : HusAero Complete zip의 dataset 폴더 (드론·비행기 등이 들어 있는 이미지, 권장)
# "camera"  : 261107/quantization.py로 카메라에서 모은 이미지 (실습과 같은 방식)
CALIBRATION = "husaero"

HUSAERO_DATASET = ROOT / "src/datasets/husaero"                       # zip의 dataset 폴더를 여기로 복사
HUSAERO_YAML = HUSAERO_DATASET / "calibration.yaml"
CAMERA_YAML = ROOT / "261107/src/datasets/calibration/calibration.yaml"


### 1. 모델 확인
if not PT_PATH.exists():
    raise SystemExit(f"HusAero-8n.pt가 없습니다: {PT_PATH}")

model = YOLO(str(PT_PATH))
print("클래스 번호:", model.names)


### 2. 캘리브레이션 yaml 준비
if CALIBRATION == "husaero":
    if not (HUSAERO_DATASET / "images/val").exists():
        raise SystemExit(
            f"{HUSAERO_DATASET}/images/val 이 없습니다.\n"
            "HusAero Complete zip의 dataset 폴더를 src/datasets/husaero 로 복사해 주세요."
        )

    # 261107/quantization.py와 같은 형식의 yaml (경로만 HusAero dataset으로)
    calibration_yaml = {
        "path": str(HUSAERO_DATASET),
        "train": "images/train",
        "val": "images/val",
        "names": model.names,
    }

    with open(HUSAERO_YAML, "w", encoding="utf-8") as file:
        yaml.safe_dump(calibration_yaml, file, sort_keys=False, allow_unicode=True)

    data_yaml = HUSAERO_YAML

else:
    if not CAMERA_YAML.exists():
        raise SystemExit(f"{CAMERA_YAML} 이 없습니다. 261107/quantization.py를 먼저 실행해 주세요.")
    data_yaml = CAMERA_YAML

print(f"캘리브레이션 데이터: {data_yaml}")


### 3. TensorRT INT8 엔진으로 변환 (04 노트북과 같은 설정)
engine_path = model.export(
    format="engine",
    imgsz=640,
    quantize=8,
    data=str(data_yaml),
    batch=1,
    dynamic=False,
    device=0,
    nms=False,
)

print(f"TensorRT Engine 생성 완료: {engine_path}")

# 파일 이름 변경 (04 노트북과 같은 방식)
int8_engine_path = Path(engine_path).replace(ENGINE_PATH)

print(f"INT8 Engine: {int8_engine_path}")
