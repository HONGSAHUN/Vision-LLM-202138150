import cv2
import numpy as np

pipeline = (
    "v4l2src device=/dev/video0 ! "
    "image/jpeg,width=1280,height=720,framerate=30/1 ! "
    "jpegdec ! videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 "
    "max-size-bytes=0 max-size-time=0 ! "
    "appsink drop=true max-buffers=1 sync=false"
)

# max-buffers = 카메라의 딜레이 줄여주는 역할

green_lower = np.array([30, 60, 90], dtype=np.uint8)
green_upper = np.array([230, 115, 180], dtype=np.uint8)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

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
    mask_enhanced_rgb = cv2.cvtColor(ball_enhanced_lab, cv2.COLOR_LAB2RGB)

    ball_mask_opened = cv2.morphologyEx(mask_enhanced, cv2.MORPH_OPEN, kernel)
    final_ball_mask = ball_mask_opened

    contours, _ = cv2.findContours(
        final_ball_mask,        
        cv2.RETR_EXTERNAL,       
        cv2.CHAIN_APPROX_SIMPLE  
    )

    if len(contours) > 0:
        # 가장 큰 contour
        largest_contour = max(contours, key=cv2.contourArea)

        # 가장 큰 contour만 채운 binary mask 생성
        largest_mask = np.zeros_like(final_ball_mask)
        cv2.drawContours(
            largest_mask,
            [largest_contour],
            -1,
            255,
            thickness=cv2.FILLED
        )


        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        center_circle = (int(x), int(y))
        radius = int(radius)
        
        # Moments를 통해 무게중심 (Centroid) 산출
        M = cv2.moments(largest_contour)
        
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = center_circle
        
        centroid = (cX, cY)
        
        # 빨간색 무게중심 기반 원
        cv2.circle(frame, centroid, radius, (0, 0, 255), 3)
        cv2.circle(frame, centroid, 5, (0, 0, 255), -1)

        cv2.imshow("hello", frame)

cap.release()
cv2.destroyAllWindows()



