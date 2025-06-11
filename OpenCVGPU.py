import cv2
import numpy as np

# 이미지 사이즈 정의
width, height = 1920, 1080

# 랜덤 노이즈 이미지 생성 (기본적으로는 CPU 사용)
cpu_img = np.random.randint(0, 256, (height, width), dtype=np.uint8)

# CUDA GPU로 전송
gpu_img = cv2.cuda_GpuMat()
gpu_img.upload(cpu_img)

# Gaussian Blur를 GPU에서 실행
gpu_blurred = cv2.cuda.createGaussianFilter(cv2.CV_8UC1, cv2.CV_8UC1, (15, 15), 0)
gpu_result = gpu_blurred.apply(gpu_img)

# 다시 CPU로 다운로드
result_img = gpu_result.download()

# 저장
cv2.imwrite("opencv_gpu_output.png", result_img)
print("✅ OpenCV GPU 기반 이미지 처리 완료")
