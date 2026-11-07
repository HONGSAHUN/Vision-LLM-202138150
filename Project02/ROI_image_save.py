import cv2
import os

from ultralytics import YOLO


YOLO_MODEL_PATH = "../src/models/YOLO/yolo11n_int8.engine"
OUTPUT_IMAGE_PATH = "../src/output/vision_output.jpg"
TEMP_IMAGE_PATH = "../src/output/vision_output_temp.jpg"


yolo = YOLO(YOLO_MODEL_PATH)

cap = cv2.VideoCapture(0)


if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()


print("q : 종료")
print("l : 가장 왼쪽 객체 이미지 저장")


while True:
    ret, frame = cap.read()

    if not ret:
        break

    # YOLO 객체 탐지
    result = yolo.predict(
        source=frame,
        conf=0.25,
        iou=0.5,
        verbose=False,
    )[0]

    # 탐지된 객체의 Bounding Box 정보 추출
    detected_objects = []

    for box in result.boxes:
        x1, y1, x2, y2 = (box.xyxy[0].cpu().tolist())

        detected_objects.append({
            "bbox": (x1, y1, x2, y2),
            "confidence": float(box.conf[0].item()),
            "class_id": int(box.cls[0].item()),
            "class_name": result.names[
                int(box.cls[0].item())
            ],
        })

    # 가장 왼쪽 객체 선택
    if detected_objects:
        selected_object = min(detected_objects, key=lambda x: x["bbox"][0])

    # 객체 탐지 결과 화면 출력
    output_frame = result.plot()

    cv2.imshow("YOLO", output_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    # 키 입력 시 ROI 이미지 저장
    if key == ord("l"):
        if not detected_objects:
            print("탐지된 객체가 없습니다.")
            continue

        # 선택한 객체의 Bounding Box 추출
        x1, y1, x2, y2 = selected_object["bbox"]

        # 이미지 크기에 맞게 좌표 보정
        height, width = frame.shape[:2]

        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(width, int(x2))
        y2 = min(height, int(y2))

        # ROI 이미지 추출
        object_image = frame[y1:y2, x1:x2].copy()

        if object_image.size == 0:
            print("객체 이미지를 추출할 수 없습니다.")
            continue

        # 임시 파일 저장
        success = cv2.imwrite(TEMP_IMAGE_PATH, object_image)

        if not success:
            print("이미지 저장에 실패했습니다.")
            continue

        # 기존 이미지 교체
        os.replace(TEMP_IMAGE_PATH, OUTPUT_IMAGE_PATH)

        print(f"이미지 저장 완료: {OUTPUT_IMAGE_PATH}")


cap.release()
cv2.destroyAllWindows()