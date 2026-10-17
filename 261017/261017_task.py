import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 불러오고, LAB변환, CLAHE 생성 후 Lightness 강화하여 이진 마스크 생성
basketball = cv2.imread('../src/images/basketball_crop.jpg')
basketball_rgb = cv2.cvtColor(basketball, cv2.COLOR_BGR2RGB)

lower_basketball_lab = np.array([70, 130, 70])
upper_basketball_lab = np.array([180, 230, 185])
 
basketball_lab = cv2.cvtColor(basketball, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(basketball_lab)
 
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

l_clahe_enhanced = clahe.apply(l)
basketball_enhanced_lab = cv2.merge((l_clahe_enhanced, a, b))
 
mask_enhanced = cv2.inRange(basketball_enhanced_lab, lower_basketball_lab, upper_basketball_lab)

# TODO: Morphological operation을 적용하여 최종 binary mask 생성
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Opening과 Closing 중 무엇이 더 나은지 비교 -> 노이즈가 더 적어서 Opening으로 결정
basketball_mask_opened = cv2.morphologyEx(mask_enhanced, cv2.MORPH_OPEN, kernel)
basketball_mask_closed = cv2.morphologyEx(mask_enhanced, cv2.MORPH_CLOSE, kernel)

plt.figure(figsize=(16,8))
plt.subplot(1,3,1), plt.imshow(mask_enhanced, cmap="gray"), plt.title("Binary Mask")
plt.subplot(1,3,2), plt.imshow(basketball_mask_opened, cmap="gray"), plt.title("Mask after Opening")
plt.subplot(1,3,3), plt.imshow(basketball_mask_closed, cmap="gray"), plt.title("Mask after Closing")

for ax in plt.gcf().axes:
    ax.axis("off")
plt.show()
    
# Opening 횟수 비교 -> iteration = 2와 3이 비슷하지만, 면적이 더 넓어보이는 2로 결정
ball_mask_opened1 = cv2.morphologyEx(mask_enhanced, cv2.MORPH_OPEN, kernel)
ball_mask_opened2 = cv2.morphologyEx(mask_enhanced, cv2.MORPH_OPEN, kernel, iterations=2)
ball_mask_opened3 = cv2.morphologyEx(mask_enhanced, cv2.MORPH_OPEN, kernel, iterations=3)

plt.figure(figsize=(16,8))
plt.subplot(2,3,2), plt.imshow(basketball_rgb), plt.title("Original Image")
plt.subplot(2,3,4), plt.imshow(ball_mask_opened1, cmap="gray"), plt.title("iteration = 1")
plt.subplot(2,3,5), plt.imshow(ball_mask_opened2, cmap="gray"), plt.title("iteration = 2")
plt.subplot(2,3,6), plt.imshow(ball_mask_opened3, cmap="gray"), plt.title("iteration = 3")

for ax in plt.gcf().axes:
    ax.axis("off")
plt.show()

# Opening x2 결과에 Closing 횟수 비교 -> 다 비슷해보이지만, 횟수가 많을수록 테두리가 조금씩 매끄러워져서 Closing x3으로 결정
ball_mask_closed1 = cv2.morphologyEx(ball_mask_opened2, cv2.MORPH_CLOSE, kernel)
ball_mask_closed2 = cv2.morphologyEx(ball_mask_opened2, cv2.MORPH_CLOSE, kernel, iterations=2)
ball_mask_closed3 = cv2.morphologyEx(ball_mask_opened2, cv2.MORPH_CLOSE, kernel, iterations=3)

plt.figure(figsize=(16,8))
plt.subplot(2,3,2), plt.imshow(ball_mask_opened2, cmap="gray"), plt.title("Opening x2")
plt.subplot(2,3,4), plt.imshow(ball_mask_closed1, cmap="gray"), plt.title("Opening x2 & Closing x1")
plt.subplot(2,3,5), plt.imshow(ball_mask_closed2, cmap="gray"), plt.title("Opening x2 & Closing x2")
plt.subplot(2,3,6), plt.imshow(ball_mask_closed3, cmap="gray"), plt.title("Opening x2 & Closing x3")

for ax in plt.gcf().axes:
    ax.axis("off")
plt.show()

final_ball_mask = ball_mask_closed3

# TODO: Binary mask로 외곽선 검출
contours, _ = cv2.findContours(
    final_ball_mask,        
    cv2.RETR_EXTERNAL,       
    cv2.CHAIN_APPROX_SIMPLE  
)

# TODO: 최종 농구공 영역 및 외곽선 선택
final_img = basketball_rgb.copy()

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

    # Binary mask를 반투명 초록색으로 overlay
    mask_color = np.zeros_like(final_img)
    mask_color[:, :, 1] = largest_mask
    alpha = 0.35
    final_img = cv2.addWeighted(final_img, 1.0, mask_color, alpha, 0)

    # Contour 그리기 (초록색 외곽선)
    cv2.drawContours(final_img, [largest_contour], -1, (0,255,0), 2)

plt.imshow(final_img)
plt.axis("off");

plt.show()
