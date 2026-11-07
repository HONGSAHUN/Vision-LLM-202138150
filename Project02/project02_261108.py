import os, time                                     # 경로, 시간(FPS, 타이머) 계산용
import json                                         # 탐지 결과 저장용 
import subprocess                                   # STT, TTS 실행용 
from collections import deque                       # 최근 위치 기록용
from concurrent.futures import ThreadPoolExecutor   # 음성, LLM 작업을 카메라와 따로 실행

import cv2
from ultralytics import YOLO
from llama_cpp import Llama


### 경로 설정 : 이 파일(Project02)의 상위 폴더 = 레포 최상위
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def path(relative):
    return os.path.join(ROOT, relative)

YOLO_MODEL_PATH = path("src/models/YOLO/husaero/HusAero-8n_int8.engine")   # HusAero-8n을 INT8로 변환 (husaero_int8.py)
GEMMA_MODEL_PATH = path("src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf")
JSON_PATH = path("src/json/vision_data.json")
TEMP_JSON_PATH = path("src/json/vision_data_tmp.json")

CONTEXT_WINDOW = 2048
MAX_TOKENS = 150

### STT / TTS 설정 (06 C-3과 동일)
MIC_DEVICE = "plughw:3,0"
AUDIO_FILE = path("src/audio/input.wav")
WHISPER_PATH = path("whisper.cpp/build-cpu/bin/whisper-cli")
WHISPER_MODEL = path("whisper.cpp/models/ggml-base.bin")
RECORD_SECONDS = 5

PIPER_PYTHON = path(".piper_venv/bin/python")
PIPER_MODEL = path("src/models/Piper/ko_KR-kss-medium.onnx")
OUTPUT_FILE = path("src/audio/response.wav")

### 탐지 / 판단 기준
TARGET_CLASSES = [0, 2, 3] # HusAero 클래스 중 drone(0), airplane(2), helicopter(3)만 탐지 (bird, kite 제외)
TYPE_NAMES = {"drone": "드론", "airplane": "비행기", "helicopter": "헬리콥터"}   # 보고용 한글 이름
CONF = 0.25                # 대상이 낮게 잡히면 낮추기
APPEAR_SECONDS = 0.5       # 이 시간 이상 연속으로 보이면 '출현'
GAP_SECONDS = 0.3          # 이보다 짧게 놓친 것은 연속으로 봄 (깜빡임 대응)
LOST_SECONDS = 3.0         # 이 시간 이상 안 보이면 '표적 상실'
HISTORY_SECONDS = 1.0      # 방향 계산에 쓰는 최근 기록 길이
MOVE_RATIO = 0.05          # 화면 크기의 5% 이상 움직여야 이동으로 판단
APPROACH_RATIO = 1.2       # bbox 면적이 1.2배 이상 커지면 접근
RECEDE_RATIO = 0.8         # 0.8배 이하로 작아지면 멀어짐
REPORT_INTERVAL = 10.0     # 추적 중 보고 간격 (초)
MAX_ASK = 3                # 피아 식별 질문 최대 횟수

### 상태 정의 (스테이트 머신)
SEARCH = "SEARCH"          # 비행체 탐색
IDENTIFY = "IDENTIFY"      # 보고 + 피아 식별 질문
TRACK = "TRACK"            # 적군(또는 미식별) 추적
FINISH = "FINISH"          # 종료 안내 후 프로그램 종료


### STT (06 C-3과 동일)
def speech_to_text():
    subprocess.run(
        [
            "pasuspender", "--",
            "arecord",
            "-D", MIC_DEVICE,
            "-f", "S16_LE",
            "-r", "16000",
            "-c", "1",
            "-d", str(RECORD_SECONDS),
            AUDIO_FILE,
        ],
        check=True,
    )

    result = subprocess.run(
        [
            WHISPER_PATH,
            "-m", WHISPER_MODEL,
            "-f", AUDIO_FILE,
            "-l", "ko",
            "--no-timestamps",
        ],
        text=True,
        capture_output=True,
        check=True,
    )

    return result.stdout.strip()


### TTS (06 C-3과 동일)
def text_to_speech(text):
    subprocess.run(
        [
            PIPER_PYTHON,
            "-m",
            "piper",

            "-m",
            PIPER_MODEL,

            "-f",
            OUTPUT_FILE,

            "--",
            text,
        ],
        check=True,
    )

    subprocess.run(
        [
            "aplay",
            OUTPUT_FILE,
        ],
        check=True,
    )


### JSON → 자연어 변환 (06 A-4와 동일. 추적 대상 정보 두 줄만 추가(136-137))
def detections_to_text(json_path):
    with open(json_path, "r", encoding="utf-8") as file:
        vision_data = json.load(file)

    objects = vision_data["objects"]

    if len(objects) == 0:
        return "현재 탐지된 객체가 없습니다."

    sentences = []

    for index, obj in enumerate(objects, start=1):
        sentence = f"{index}번 객체는 {obj['class']}이며, confidence는 {obj['confidence']:.2f}입니다."

        sentences.append(sentence)

    target = vision_data["target"]
    sentences.append(f"추적 대상의 종류는 {target['type']}이며, {target['position']}에 있고, {target['direction']}, {target['distance']}입니다.")

    return "\n".join(sentences)


### Dictionary → JSON 저장 (06 A-7과 동일 : 임시 파일에 쓴 뒤 교체)
def save_json(vision_dict):
    with open(TEMP_JSON_PATH, "w", encoding="utf-8") as file:
        json.dump(vision_dict, file, ensure_ascii=False, indent=4)

    os.replace(TEMP_JSON_PATH, JSON_PATH)


### 화면을 3x3으로 나눠 위치를 글로 표현
def position_text(cx, cy, width, height):
    col = ["왼쪽", "가운데", "오른쪽"][min(2, int(cx / width * 3))]
    row = ["위", "가운데", "아래"][min(2, int(cy / height * 3))]

    if col == "가운데" and row == "가운데":
        return "화면 중앙"
    if row == "가운데":
        return f"화면 {col}"
    if col == "가운데":
        return f"화면 {row}쪽"
    return f"화면 {col} {row}"


### 최근 기록의 처음과 끝을 비교해 이동 방향과 접근 여부 판단
def motion_text(history, width, height):
    if len(history) < 2:
        return "이동 판단 불가", "거리 변화 판단 불가"

    _, cx0, cy0, area0 = history[0]
    _, cx1, cy1, area1 = history[-1]

    dx = (cx1 - cx0) / width
    dy = (cy1 - cy0) / height          # 이미지 좌표는 아래로 갈수록 y가 커짐

    parts = []
    if dx > MOVE_RATIO:
        parts.append("오른쪽")
    elif dx < -MOVE_RATIO:
        parts.append("왼쪽")
    if dy > MOVE_RATIO:
        parts.append("아래쪽")
    elif dy < -MOVE_RATIO:
        parts.append("위쪽")

    direction = (" ".join(parts) + "으로 이동 중") if parts else "거의 정지"

    ratio = area1 / area0 if area0 > 0 else 1.0
    if ratio >= APPROACH_RATIO:
        distance = "접근 중"
    elif ratio <= RECEDE_RATIO:
        distance = "멀어지는 중"
    else:
        distance = "거리 변화 없음"

    return direction, distance


### Gemma 호출 (06 A-5 / B-2 형식)
def ask_gemma(system_prompt, user_text, max_tokens, temperature):
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )

    return response["choices"][0]["message"]["content"].strip()


REPORT_PROMPT = """
Instruction:
주어진 객체 탐지 정보를 바탕으로 미확인 비행체의 종류, 위치, 이동 방향을 보고하시오.
{task}

Constraint:
탐지 정보에 없는 내용을 추측하지 마시오.
거리는 추정값이므로 숫자로 말하지 마시오.

Output Format:
한국어 두 문장 이내. 특수문자 없이 평문.
"""

CLASSIFY_PROMPT = """
Instruction:
사용자의 발화가 비행체를 아군으로 식별했는지, 적군으로 식별했는지 판단하시오.

Constraint:
반드시 다음 세 가지 중 하나만 대답하시오.
아군
적군
불명

적, 적기, 적군기, 적 드론, 적성 같은 표현은 적군으로 대답하시오.
아군기, 우군, 우리 편 같은 표현은 아군으로 대답하시오.
"적군 아니야", "아군 아니야"처럼 부정만 하는 표현은 불명으로 대답하시오.
판단이 어렵거나 관련 없는 발화면 불명으로 대답하시오.
다른 설명이나 문장을 추가하지 마시오.

Output Format:
한 단어.
"""


### 음성 작업 1 : 보고(또는 재질문) → 녹음 → 피아 식별. 결과는 "아군" / "적군" / "불명"
def identify_job(vision_text, first):
    if first:
        prompt = REPORT_PROMPT.format(task="마지막 문장에서 아군인지 적군인지 식별을 요청하시오.")
        report = ask_gemma(prompt, vision_text, MAX_TOKENS, 0.3)
        print(f"\n[Gemma 보고] {report}")
        text_to_speech(report)
    else:
        text_to_speech("식별하지 못했습니다. 아군인지 적군인지 다시 말씀해 주세요.")

    print(f"\n[STT] {RECORD_SECONDS}초간 말씀하세요.")
    question = speech_to_text()
    print(f"[STT 결과] {question}")

    if not question or "[BLANK_AUDIO]" in question:
        return "불명"

    answer = ask_gemma(CLASSIFY_PROMPT, question, 10, 0.0)

    for label in ("아군", "적군"):     # 형식을 어겨도 라벨만 골라냄
        if label in answer:
            return label
    return "불명"


### 음성 작업 2 : 추적 중 정기 보고
def track_report_job(vision_text):
    prompt = REPORT_PROMPT.format(task="추적 중인 비행체이므로 식별 요청은 하지 마시오.")
    report = ask_gemma(prompt, vision_text, MAX_TOKENS, 0.3)
    print(f"\n[Gemma 보고] {report}")
    text_to_speech(report)
    return "done"


### 음성 작업 3 : 정해진 문장 안내
def speak_job(text):
    print(f"\n[안내] {text}")
    text_to_speech(text)
    return "done"


### 끝난 작업의 결과 꺼내기 (작업 중 오류가 나도 프로그램이 멈추지 않도록)
def take_result(job):
    try:
        return job.result()
    except Exception as error:
        print(f"\n[음성 작업 오류] {error}")
        return "오류"


### 화면 글자 (검은 외곽선 + 흰 글자)
def put_text(image, text, position, color=(255, 255, 255)):
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


### 실행 전 점검 : 필요한 파일이 없으면 시작하지 않음
for name, file_path in [
    ("YOLO 엔진 (Project02/husaero_int8.py로 먼저 변환)", YOLO_MODEL_PATH),
    ("Gemma 모델", GEMMA_MODEL_PATH),
    ("whisper-cli", WHISPER_PATH),
    ("Whisper 모델", WHISPER_MODEL),
    ("Piper 파이썬", PIPER_PYTHON),
    ("Piper 음성", PIPER_MODEL),
]:
    if not os.path.exists(file_path):
        raise SystemExit(f"{name} 파일이 없습니다: {file_path}")

os.makedirs(os.path.dirname(JSON_PATH), exist_ok=True)
os.makedirs(os.path.dirname(AUDIO_FILE), exist_ok=True)


### 모델 불러오기 (06 A 형식)
yolo = YOLO(YOLO_MODEL_PATH)

llm = Llama(
    model_path=GEMMA_MODEL_PATH,
    n_gpu_layers=-1,
    n_ctx=CONTEXT_WINDOW,
    verbose=False,
)

# 음성·LLM 작업 담당 (일꾼 1명 → 작업은 한 번에 하나씩, 맡긴 순서대로 실행 → 안내 음성과 보고 음성이 겹치지 않음)
worker = ThreadPoolExecutor(max_workers=1)


### GStreamer pipeline으로 카메라 열기 (Project01과 동일)
pipeline = (
    "v4l2src device=/dev/video0 ! "
    "image/jpeg,width=1280,height=720,framerate=30/1 ! "
    "jpegdec ! videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 "
    "max-size-bytes=0 max-size-time=0 ! "
    "appsink drop=true max-buffers=1 sync=false"
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    raise SystemExit("카메라를 열 수 없습니다.")


### 프레임 사이에 기억할 변수
state = SEARCH             # 현재 상태
job = None                 # 진행 중인 음성 작업
ask_count = 0              # 식별 질문 횟수
identity = "미식별"        # 식별 결과
last_report = 0.0          # 마지막 추적 보고 시각

history = deque()          # 최근 (시각, 중심x, 중심y, 면적)
first_seen = None          # 연속 탐지가 시작된 시각
last_seen = None           # 마지막으로 탐지된 시각
displayed_fps = 0.0

print("q : 종료")
print(f"[STATE] {state}")


while True:
    start_time = time.perf_counter()
    now = time.time()

    # 1. 프레임 읽기 (06 A-6 형식 동일)
    ret, frame = cap.read()
    key = cv2.waitKey(1) & 0xFF

    if not ret:
        break
    if key == ord("q"):
        break

    height, width = frame.shape[:2]

    # 2. YOLO 탐지 (드론, 비행기, 헬리콥터만 → TARGET_CLASSES)
    results = yolo.predict(
        source=frame,
        conf=CONF,
        iou=0.5,
        classes=TARGET_CLASSES,
        verbose=False,
    )

    result = results[0]

    # 3. 탐지 결과를 Dictionary 형태로 변환 (06 A-2와 동일)
    objects = []

    for box in result.boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        x1, y1, x2, y2 = (box.xyxy[0].cpu().tolist())

        objects.append(
            {
                "class": result.names[class_id],
                "confidence": round(confidence, 3),
                "bbox": {
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                }
            }
        )

    vision_dict = {
        "image_width": width,
        "image_height": height,
        "objects": objects,
    }

    # 4. 추적 대상 정리 : confidence가 가장 높은 비행체 하나
    detected = len(objects) > 0

    if detected:
        target = max(objects, key=lambda obj: obj["confidence"])
        bbox = target["bbox"]
        cx = (bbox["x1"] + bbox["x2"]) / 2
        cy = (bbox["y1"] + bbox["y2"]) / 2
        area = (bbox["x2"] - bbox["x1"]) * (bbox["y2"] - bbox["y1"])

        history.append((now, cx, cy, area))
        last_seen = now
        if first_seen is None:
            first_seen = now

        direction, distance = motion_text(history, width, height)
        vision_dict["target"] = {
            "type": TYPE_NAMES.get(target["class"], target["class"]),
            "position": position_text(cx, cy, width, height),
            "direction": direction,
            "distance": distance,
        }

    elif last_seen is not None and now - last_seen > GAP_SECONDS:
        first_seen = None          # 잠깐 놓친 게 아니면 연속 탐지 끊김

    while history and now - history[0][0] > HISTORY_SECONDS:
        history.popleft()

    appeared = detected and first_seen is not None and now - first_seen >= APPEAR_SECONDS
    lost = last_seen is None or now - last_seen > LOST_SECONDS

    # 5. 끝난 음성 작업이 있으면 결과 받기
    job_result = None
    if job is not None and job.done():
        job_result = take_result(job)
        job = None

    # 6. 스테이트 머신 : 상태별 판단(next_state 결정) + 상태에 머무는 동안의 동작
    next_state = state

    if state == SEARCH:
        if appeared:
            next_state = IDENTIFY

    elif state == IDENTIFY:
        if job_result is not None:
            print(f"[식별 결과] {job_result}")

            if job_result == "아군":
                identity = "아군"
                next_state = FINISH
            elif job_result == "적군":
                identity = "적군"
                next_state = TRACK
            elif ask_count >= MAX_ASK:
                identity = "미식별"
                next_state = TRACK
            else:
                ask_count += 1
                job = worker.submit(identify_job, "", False)       # 다시 질문

    elif state == TRACK:
        if lost:
            next_state = SEARCH
        elif job is None and detected and now - last_report >= REPORT_INTERVAL:
            save_json(vision_dict)
            job = worker.submit(track_report_job, detections_to_text(JSON_PATH))
            last_report = now

    elif state == FINISH:
        if job is None:            # 종료 안내가 끝나면 종료
            break

    # 7. 상태 전환 : state 가 바뀌는 곳은 여기뿐, 바뀐 직후 새 상태에 들어갈 때 할 일을 딱 한 번 실행.
    if next_state != state:
        print(f"\n[STATE] {state} → {next_state}")
        state = next_state

        if state == IDENTIFY: # 질문 횟수 1로 초기화 → JSON 저장 → 자연어 변환 → 첫 보고, 질문 작업 제출
            ask_count = 1
            save_json(vision_dict)
            job = worker.submit(identify_job, detections_to_text(JSON_PATH), True)

        elif state == TRACK: # 보고 타이머 초기화 → 적군, 미식별 표적 판단 안내
            last_report = now
            job = worker.submit(speak_job, f"{identity} 표적으로 판단합니다. 추적을 계속합니다.")

        elif state == SEARCH: # 식별 결과 초기화 → 표적 놓침 안내
            identity = "미식별"
            job = worker.submit(speak_job, "표적을 놓쳤습니다. 탐색 모드로 전환합니다.")

        elif state == FINISH: # 아군 식별 안내
            job = worker.submit(speak_job, "아군으로 식별했습니다. 추적을 종료합니다.")

    # 8. 화면 표시 (FPS 계산은 04 실습 파일과 같은 지수 평균 방식 적용)
    output_frame = result.plot()

    elapsed_time = time.perf_counter() - start_time
    current_fps = 1.0 / elapsed_time if elapsed_time > 0 else 0.0

    if displayed_fps == 0:
        displayed_fps = current_fps
    else:
        displayed_fps = 0.9 * displayed_fps + 0.1 * current_fps

    state_color = {SEARCH: (255, 255, 255), IDENTIFY: (0, 255, 255), TRACK: (0, 0, 255), FINISH: (0, 255, 0)}[state]
    id_text = {"아군": "FRIENDLY", "적군": "HOSTILE", "미식별": "UNKNOWN"}[identity]   # putText는 한글 표시 불가

    put_text(output_frame, f"STATE : {state}", (20, 40), state_color)
    put_text(output_frame, f"FPS : {displayed_fps:.1f}", (20, 75))
    if state == TRACK:
        put_text(output_frame, f"ID : {id_text}", (20, 110), state_color)
    if job is not None:
        put_text(output_frame, "VOICE / LLM ...", (20, height - 30), (0, 255, 255))

    cv2.imshow("Project02 - Aerial Detection", output_frame)


### 종료 및 정리
print("\n프로그램을 종료합니다. (진행 중인 음성 작업이 있으면 끝날 때까지 기다립니다)")
cap.release()
cv2.destroyAllWindows()
worker.shutdown(wait=True, cancel_futures=True)
