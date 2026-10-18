import cv2
import numpy as np
from numpy.linalg import inv

pipeline = (
    "v4l2src device=/dev/video0 ! "
    "image/jpeg,width=1280,height=720,framerate=30/1 ! "
    "jpegdec ! videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 "
    "max-size-bytes=0 max-size-time=0 ! "
    "appsink drop=true max-buffers=1 sync=false"
)

def KalmanFilter(mu_prev, sigma_prev, z):
    mu_bar = A_t.dot(mu_prev)
    sigma_bar = A_t.dot(sigma_prev).dot(A_t.transpose()) + R_t
    if z is None:
        return mu_bar, sigma_bar
    else:
        K_t = sigma_bar.dot(C_t.transpose()).dot(inv(C_t.dot(sigma_bar).dot(C_t.transpose()) + Q_t))
        mu = mu_bar + K_t.dot(z - C_t.dot(mu_bar))
        sigma = (np.identity(2) - K_t.dot(C_t)).dot(sigma_bar)
        return mu, sigma

# Kalman filter 변수 정의
A_t = np.array([[1, 1], [0, 1]])
G = np.array([[0.5], [1]])
R_t = G.dot(G.transpose())
C_t = np.array([[1, 0]])
Q_t = np.array([[1]])
mu_t = np.array([[0, 0], [0, 0]])
sigma_t = np.array([[0, 0], [0, 0]])


green_lower = np.array([30, 60, 90], dtype=np.uint8)
green_upper = np.array([230, 115, 180], dtype=np.uint8)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

# TODO: Kalman Filter를 활용하여 객체 추적

# TODO: 탐지된 객체의 중심과 반지름을 구해 프레임 위에 overlay
# TODO: 예측된 객체의 중심과 반지름을 구해 프레임 위에 다른 색으로 overlay


# 객체 최초 검출 여부 확인용 boolean
found = False

while True:
    ret, frame = cap.read()

    if not ret:
        break
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_blur = cv2.GaussianBlur(frame, (7, 7), 0)
    frame_blur_lab = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(frame_blur_lab)
    l_clahe_enhanced = clahe.apply(l)
    ball_enhanced_lab = cv2.merge((l_clahe_enhanced, a, b))

    mask_enhanced = cv2.inRange(ball_enhanced_lab, green_lower, green_upper)
    # mask_enhanced_rgb = cv2.cvtColor(ball_enhanced_lab, cv2.COLOR_LAB2RGB)

    ball_mask_opened = cv2.morphologyEx(mask_enhanced, cv2.MORPH_OPEN, kernel)
    final_ball_mask = ball_mask_opened

    contour_lst, _ = cv2.findContours(
        final_ball_mask,        
        cv2.RETR_EXTERNAL,       
        cv2.CHAIN_APPROX_SIMPLE  
    )

    if len(contour_lst) > 0:
        # 가장 큰 contour 선택
        contour = max(contour_lst, key=cv2.contourArea)

        # 최소 외접원 반지름
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center_circle = (int(x), int(y))
        radius = int(radius)
        # 무게중심
        center = (int(x), int(y))
        M = cv2.moments(contour)
        
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = center_circle

        centroid = (cX, cY)

        # 검출된 객체에 파란 원 overlay
        cv2.circle(frame, centroid, int(radius), (255, 0, 0), 3)
        cv2.circle(frame, centroid, 5, (255, 0, 0), -1)

        # 객체 최초 검출
        if not found:
            found = True

    #최초 검출 이후 Kalman Filter 적용
    #측정값 사용 (visible) -> Prediction & Update
    if found and (len(contour_lst) > 0):
        mu_t, sigma_t = KalmanFilter(mu_t, sigma_t, np.array([list(center)]))
        x_bel = mu_t[0][0]
        y_bel = mu_t[0][1]

    # 측정값 미사용 (occluded) -> Prediction
    elif found and (len(contour_lst) <= 0):
        mu_t, sigma_t = KalmanFilter(mu_t, sigma_t, None)
        x_bel = mu_t[0][0]
        y_bel = mu_t[0][1]

    # 예측한 객체 위치에 노란 원 overlay
    if found:
        cv2.circle(frame, (int(x_bel), int(y_bel)), int(radius), (0, 255, 255), 3)
        cv2.circle(frame, (int(x_bel), int(y_bel)), 5, (0, 255, 255), -1)

    cv2.imshow("hello", frame)

cap.release()
cv2.destroyAllWindows()