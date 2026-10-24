import time                 # FPS 계산용
import cv2                  # 영상 처리용
import numpy as np          # 벡터, 행렬 계산용
import mediapipe as mp      # 손 인식용
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

### finger_tip.py와 동일
def calculate_angle(p1, p2, p3):
    vector1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
    vector2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])

    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0 # 길이가 0이라 0으로 나누게 되는 것을 방지

    cosine = np.dot(vector1, vector2) / (magnitude1 * magnitude2)
    cosine = np.clip(cosine, -1.0, 1.0) # 계산 오차로 인한 범위 초과 방지

    return np.degrees(np.arccos(cosine))

### 두 벡터 사이 각도 계산하는 함수 (조준 방향과 공 방향 비교용, 위 함수와 같은 원리로 구성)
def vector_angle(vector1, vector2):
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    if magnitude1 == 0 or magnitude2 == 0:
        return 180.0

    cosine = np.dot(vector1, vector2) / (magnitude1 * magnitude2)
    cosine = np.clip(cosine, -1.0, 1.0)

    return np.degrees(np.arccos(cosine))

# GStreamer pipeline으로 열기
pipeline = (
    "v4l2src device=/dev/video0 ! "
    "image/jpeg,width=1280,height=720,framerate=30/1 ! "
    "jpegdec ! videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 "
    "max-size-bytes=0 max-size-time=0 ! "
    "appsink drop=true max-buffers=1 sync=false"
)   # 오래된 프레임을 버리고, 최신 1장만 유지

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

### 공 검출 설정 (기존 내용과 동일)
green_lower = np.array([30, 60, 90], dtype=np.uint8)
green_upper = np.array([230, 115, 180], dtype=np.uint8)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

### 손 검출 설정 (기존 내용과 동일)
base_option = python.BaseOptions(model_asset_path="src/models/MediaPipe/hand_landmarker.task")
options = vision.HandLandmarkerOptions(base_options=base_option, num_hands=2)
hand_detector = vision.HandLandmarker.create_from_options(options)
connections = vision.HandLandmarksConnections.HAND_CONNECTIONS
finger_tips = (4, 8, 12, 16, 20)

angle_threshold = 160 # 손가락 펼쳐짐 여부 판단하는 각도 임계값

# 각 손가락의 각도를 계산할 landmark index
finger_angle_points = (
    (1, 2, 3),
    (5, 6, 7),
    (9, 10, 11),
    (13, 14, 15),
    (17, 18, 19),
)

### OpenCV Kalman Filter 설정 (realtime_object_tracking_02.py와 동일)
# 상태 벡터 : [x, y, vx, vy] -> 위치와 속도
# 측정 벡터 : [x, y] -> 위치
kalman = cv2.KalmanFilter(4, 2)

# kalman.transitionMatrix(F) : 다음위치 = 지금위치 + 속도, 속도는 그대로(등속 운동) 가정 = A_t
kalman.transitionMatrix = np.array(     
    [
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ],
    dtype=np.float32,
)
# kalman.measurementMatrix(H) : 측정 행렬. 상태 중 위치만 측정 된다. = C_t
kalman.measurementMatrix = np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ],
    dtype=np.float32,
)

# kalman.processNoiseCov(Q) : 프로세스 노이즈. 예측 규칙이 얼마나 틀릴 수 있는지  = R_t
kalman.processNoiseCov = np.array(
    [
        [1e-2, 0, 0, 0],
        [0, 1e-2, 0, 0],
        [0, 0, 5e-2, 0],
        [0, 0, 0, 5e-2],
    ],
    dtype=np.float32,
)

# kalman.measurementNoiseCov(R) : 측정 노이즈. 측정값이 얼마나 흔들리는지 = Q_t
# kalman.errorCovPost(P) : 추정 오차 공분산. 현재 추정이 얼마나 불확실한지 = sigma
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
kalman.errorCovPost = np.eye(4, dtype=np.float32)

# 실행 시 탄 속도와 탄약 수 입력 (엔터만 누르면 디폴트 30, 탄속은 60 이하로 설정하도록 변경)
bullet_speed = input("bullet speed(default = 30, maximum = 60) : ")
ammo = input("ammo(default = 30) : ")

# 요격 설정
bullet_speed = int(bullet_speed) if bullet_speed else 30     # 탄 속도 (픽셀 / 프레임)
if bullet_speed > 60: raise SystemExit("탄속이 너무 빠릅니다.")
ammo = int(ammo) if ammo else 30
max_lead = 30           # 최대 몇 프레임 앞까지 예측할지

found = False           # 객체 최초 검출 여부
miss_count = 0          # 공이 안 보인 프레임 수
bullet = None           # 날아가는 탄 [x, y, vx, vy] (없으면 None)
prev_fire = False       # 이전 프레임의 발사 자세 여부 (한 번 접을 때 한 발만 쏘기 위함)
hit = 0
shot = 0
result_text = ""
result_timer = 0
view = 3                # 1: 원본, 2: 마스크, 3: 결과
prev_time = time.time()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # 1ms 동안 키를 확인하고, imshow 창을 갱신
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key in (ord("1"), ord("2"), ord("3")):
        view = key - ord("0")

    ### 이미지 조정 : 처리 속도를 위해 가로 960으로 축소, 이후 좌우 반전
    frame = cv2.resize(frame, (960, int(frame.shape[0] * 960 / frame.shape[1])))
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    original = frame.copy() # 그림을 그리기 전 원본 복사 -> 손 검출과 1번 화면에 사용

    ### 공 검출 : 기존 방식과 동일
    frame_blur = cv2.GaussianBlur(frame, (7, 7), 0)
    frame_blur_lab = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(frame_blur_lab)
    l_clahe_enhanced = clahe.apply(l)
    ball_enhanced_lab = cv2.merge((l_clahe_enhanced, a, b))
    # 밝기 채널에만 CLAHE를 적용해서 색은 건드리지 않고 어두운 부분 포착

    mask_enhanced = cv2.inRange(ball_enhanced_lab, green_lower, green_upper)

    ball_mask_opened = cv2.morphologyEx(mask_enhanced, cv2.MORPH_OPEN, kernel, iterations=2)
    final_ball_mask = cv2.morphologyEx(ball_mask_opened, cv2.MORPH_CLOSE, kernel, iterations=3)
    # Opening으로 작은 흰 점들을 지우고, Closing으로 공 안쪽 구멍을 메움


    # 마스크에서 흰 덩어리들의 윤곽선을 찾아서 목록으로 돌려주는 줄 -> 여기서 가장 큰 것을 공으로 고름.
    contour_lst, _ = cv2.findContours(
        final_ball_mask,            # 윤곽을 찾을 이미지    
        cv2.RETR_EXTERNAL,          # 어떤 윤곽을 찾을지 (바깥 테두리 하나만)
        cv2.CHAIN_APPROX_SIMPLE     # 테두리 점 저장 방식 (일직선 구간은 양 끝점만 저장 -> 메모리와 계산량 감소)
    )

    detected = False

    if len(contour_lst) > 0:
        contour = max(contour_lst, key=cv2.contourArea)

        # 너무 작은 contour는 노이즈로 보고 무시
        if cv2.contourArea(contour) > 100:
            detected = True

            # 최소 외접원 반지름
            (x, y), radius = cv2.minEnclosingCircle(contour)
            radius = int(radius)

            # 무게중심
            M = cv2.moments(contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroid = (cX, cY)
            
            # 검출된 객체에 파란 원 overlay
            cv2.circle(frame, centroid, radius, (255, 0, 0), 3)

            # 객체 최초 검출 (최초 state 저장)
            # predict()는 statePost를 기준으로 계산
            if not found:
                kalman.statePre = np.array([[cX], [cY], [0.0], [0.0]], dtype=np.float32)
                kalman.statePost = np.array([[cX], [cY], [0.0], [0.0]], dtype=np.float32)
                kalman.errorCovPost = np.eye(4, dtype=np.float32)
                found = True

    ### kalman Filter로 공 추적

    # 보이고 탐지가 된다면 -> 예측하고, 측정값으로 보정
    if found and detected:
        z = np.array([[np.float32(cX)], [np.float32(cY)]])
        kalman.predict()
        state = kalman.correct(z)
        miss_count = 0

    # 추적은 하고 있는데, 공이 보이지 않음(가려짐) -> 예측만 해서 직전 속도로 이어감
    elif found:
        state = kalman.predict()
        miss_count += 1

        # 30프레임 동안 안 보이면 추적 포기
        if miss_count > 30:
            found = False

    # kalman Filter가 추정한 공 위치를 꺼내서, 화면에 노란 원으로 그림 (추적 중일 때만 실행)
    if found:
        x_bel = state[0, 0]
        y_bel = state[1, 0]

        # 추정한 객체 위치에 노란 원 overlay
        cv2.circle(frame, (int(x_bel), int(y_bel)), radius, (0, 255, 255), 2)
        cv2.circle(frame, (int(x_bel), int(y_bel)), 4, (0, 255, 255), -1)

    ### 손 검출 : finger_tip.py와 동일 방식
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB) # MediaPipe는 RGB 사용
    result = hand_detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=original_rgb))

    # 좌우 반전된 화면을 기준으로 왼손과 오른손 정보 변경
    labels = ["LEFT" if handedness[0].category_name == "Right" else "RIGHT" for handedness in result.handedness]

    total_finger_count = 0
    for hand in result.hand_landmarks:
        for point1_idx, point2_idx, point3_idx in finger_angle_points:
            angle = calculate_angle(
                hand[point1_idx],
                hand[point2_idx],
                hand[point3_idx],
            )

            if angle >= angle_threshold:
                total_finger_count += 1

    # 화면 좌측 상단에 손 개수와 펼친 손가락 개수 표시
    cv2.putText(frame, f"Hands: {len(result.hand_landmarks)}", (20,35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 5)
    cv2.putText(frame, f"Hands: {len(result.hand_landmarks)}", (20,35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.putText(frame, f"Fingers: {total_finger_count}", (20,70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 5)
    cv2.putText(frame, f"Fingers: {total_finger_count}", (20,70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    # 화면 우측 상단에 왼손/오른손/양손 여부 표시
    handedness_text = " / ".join(labels)
    text_size = cv2.getTextSize(handedness_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    text_x = w - text_size[0] - 20
    cv2.putText(frame, handedness_text, (text_x, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
    cv2.putText(frame, handedness_text, (text_x, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # 탐지 결과의 각 손마다 선과 점 그리기
    for hand in result.hand_landmarks:
        h, w = frame.shape[:2]
        points = [(int(p.x * w), int(p.y * h)) for p in hand]

        for c in connections:
            cv2.line(frame, points[c.start], points[c.end], (0, 255, 0), 2)

        for i, point in enumerate(points):
            color = (0, 0, 255) if i in finger_tips else (255, 0, 0)
            cv2.circle(frame, point, 6 if i in finger_tips else 4, color, -1)

    ### 손가락별 펴짐 리스트, 검지 방향 벡터 파악, 조준 및 발사 자세 판단

    # 매 프레임마다 조준 안 함, 발사 안 함으로 먼저 정해둠
    aiming = False
    fire = False

    # 손이 하나도 없으면 len = 0 -> aiming, fire = False로 남고 건너뜀
    if len(result.hand_landmarks) > 0:
        hand = result.hand_landmarks[0]
        points = [(int(p.x * w), int(p.y * h)) for p in hand]

        # 손가락마다 펴졌는지 확인 (True / False)
        finger_up = []
        for point1_idx, point2_idx, point3_idx in finger_angle_points:
            angle = calculate_angle(hand[point1_idx], hand[point2_idx], hand[point3_idx])
            finger_up.append(angle >= angle_threshold)

        # 검지가 펴져 있으면 조준 상태, 이때 엄지를 접으면 발사 (8: 검지 끝, 5: 검지 뿌리)
        aim_start = np.array(points[8], dtype=np.float32)
        aim_dir = np.array(points[8], dtype=np.float32) - np.array(points[5])

        if finger_up[1] and np.linalg.norm(aim_dir) > 0:
            aiming = True
            aim_dir = aim_dir / np.linalg.norm(aim_dir)
            fire = not finger_up[0]

    ### 선행 조준 : 탄이 도착할 때의 공의 위치를 예측하고 LOCK 판정
    lock = False

    if found and aiming:
        # 탄이 공까지 가는 데 걸리는 프레임 수 = 거리 / 탄 속도
        distance = np.linalg.norm(np.array([x_bel, y_bel]) - aim_start)
        k = int(np.clip(distance / bullet_speed, 1, max_lead))

        # k프레임 뒤 공 위치 구하기 -> 속도 상태를 이용해서 미래를 예측하는데 사용
        future = kalman.statePost.copy()
        for i in range(k):
            future = kalman.transitionMatrix.dot(future)
        x_fut = future[0, 0]
        y_fut = future[1, 0]

        # 내가 가리키는 방향과 공이 있을 방향 사이 각도를 구하고, 그 거리에서 공이 차지하는 각도보다 작으면 LOCK
        target_dir = np.array([x_fut, y_fut]) - aim_start
        aim_error = vector_angle(aim_dir, target_dir)
        lock_angle = np.degrees(np.arctan(radius * 1.5 / max(np.linalg.norm(target_dir), 1)))
        lock = aim_error < lock_angle

        # 예측 위치에 십자선 표시
        color = (0, 255, 0) if lock else (255, 255, 0)
        cv2.drawMarker(frame, (int(x_fut), int(y_fut)), color, cv2.MARKER_CROSS, 24, 2)
        cv2.circle(frame, (int(x_fut), int(y_fut)), int(radius * 1.5), (0, 255, 0) if lock else (255, 255, 0), 2)
        cv2.circle(frame, (int(x_fut), int(y_fut)), 20, (0, 255, 0) if lock else (255, 255, 0), 2)
        if lock:
            cv2.putText(frame, "LOCK", (int(x_fut) - 30, int(y_fut) - radius - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 조준선 그리기 (검지 끝에서 1000), 락이면 초록, 아니면 빨강
    if aiming:
        aim_end = aim_start + aim_dir * 1000
        color = (0, 255, 0) if lock else (0, 0, 255)
        cv2.line(frame, (int(aim_start[0]), int(aim_start[1])), (int(aim_end[0]), int(aim_end[1])), color, 2)

    ### 발사 및 명중 판정
    # 네 조건이 모두 맞아야 사격 가능 : 엄지 접힘, 직전엔 안 접힘, 조준 중, 날아가는 탄 없음
    if fire and not prev_fire and aiming and bullet is None:
        bullet = [aim_start[0], aim_start[1], aim_dir[0] * bullet_speed, aim_dir[1] * bullet_speed]
        shot += 1
    prev_fire = fire

    if bullet is not None:
        # 탄 이동
        bullet[0] += bullet[2]
        bullet[1] += bullet[3]
        cv2.circle(frame, (int(bullet[0]), int(bullet[1])), 8, (0, 0, 0), -1)
        cv2.circle(frame, (int(bullet[0]), int(bullet[1])), 6, (255, 255, 255), -1)

        # bullet과 그 프레임의 공 위치(노란 원)이 가까우면 명중 -> 십자를 겨누어야 맞음
        if found and np.linalg.norm(np.array([bullet[0] - x_bel, bullet[1] - y_bel])) < radius:
            hit += 1
            result_text = "HIT!"
            result_timer = 20
            bullet = None

        # 화면 밖으로 나가면 빗나감
        elif not (0 <= bullet[0] < w and 0 <= bullet[1] < h):
            result_text = "MISS"
            result_timer = 20
            bullet = None

    if result_timer > 0:
        color = (0, 255, 0) if result_text == "HIT!" else (0, 0, 255)
        cv2.putText(frame, result_text, (w // 2 - 60, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.6, color, 4)
        result_timer -= 1

    ### 화면 정보 표시 : 점수, FPS, 남은 탄약
    now = time.time()
    fps = 1 / (now - prev_time)
    prev_time = now
    cv2.putText(frame, f"Score : {hit} / {shot}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
    cv2.putText(frame, f"Score : {hit} / {shot}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS : {fps:.1f}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
    cv2.putText(frame, f"FPS : {fps:.1f}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"AMMO : {ammo - shot}", (20, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
    cv2.putText(frame, f"AMMO : {ammo - shot}", (20, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    ### 게임 종료 판정 : 목표 명중 수 달성 또는 탄약 소진 경우
    end_text = ""
    if hit >= ammo / 2:
        end_text = "MISSION COMPLETE"
        end_color = (0, 255, 0)
    elif shot >= ammo and bullet is None:
        end_text = "NO AMMO"
        end_color = (0, 0, 255)

    if end_text != "":
        #화면 가운데에 종료 메시지 (검은 테두리 + 색 글자)
        text_size = cv2.getTextSize(end_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 4)[0]
        text_x = (w - text_size[0]) // 2
        text_y = h // 2 + 60
        cv2.putText(frame, end_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 8)
        cv2.putText(frame, end_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, end_color, 4)

        info = f"Score: {hit} / {shot}  (press any key)"
        cv2.putText(frame, info, (text_x, text_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
        cv2.putText(frame, info, (text_x, text_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Interceptor", frame)
        print(end_text, f"{hit} / {shot}")
        cv2.waitKey(0)
        break

    if view == 1:
        cv2.imshow("Interceptor", original)
    elif view == 2:
        cv2.imshow("Interceptor", final_ball_mask)
    else:
        cv2.imshow("Interceptor", frame)
      
hand_detector.close()
cap.release()
cv2.destroyAllWindows()
