import cv2
import numpy as np

# 512x512 파란 배경 이미지 생성
img = np.zeros((512, 512, 3), dtype=np.uint8)
img[:] = (255, 0, 0)  # BGR → 파란색

# 흰 원 그리기 (중심, 반지름, 색상, 두께)
cv2.circle(img, center=(256, 256), radius=100, color=(255, 255, 255), thickness=-1)

cv2.imwrite("circle_on_blue.png", img)
