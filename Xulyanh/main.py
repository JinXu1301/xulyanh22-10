import cv2
import numpy as np
import matplotlib.pyplot as plt


# Đọc ảnh
image_path = 'Anh1.jpg'  # Thay đổi đường dẫn tới ảnh của bạn
image = cv2.imread(image_path)

# 1. Ảnh âm tính
negative_image = 255 - image

# 2. Tăng độ tương phản (Sử dụng CLAHE)
def increase_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    contrast_image = cv2.cvtColor(limg, cv2.COLOR_Lab2BGR)
    return contrast_image

contrast_image = increase_contrast(image)

# 3. Biến đổi log
def log_transform(image):
    c = 255 / np.log(1 + np.max(image))
    log_image = c * (np.log(image + 1))
    return np.array(log_image, dtype=np.uint8)

log_image = log_transform(image)

# 4. Cân bằng histogram
def histogram_equalization(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized_image = cv2.equalizeHist(gray)
    return cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2BGR)

equalized_image = histogram_equalization(image)

# Hiển thị kết quả
plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1), plt.imshow(cv2.cvtColor(negative_image, cv2.COLOR_BGR2RGB)), plt.title('Ảnh âm tính')
plt.subplot(2, 2, 2), plt.imshow(cv2.cvtColor(contrast_image, cv2.COLOR_BGR2RGB)), plt.title('Tăng độ tương phản')
plt.subplot(2, 2, 3), plt.imshow(cv2.cvtColor(log_image, cv2.COLOR_BGR2RGB)), plt.title('Biến đổi log')
plt.subplot(2, 2, 4), plt.imshow(cv2.cvtColor(equalized_image, cv2.COLOR_BGR2RGB)), plt.title('Cân bằng Histogram')

plt.tight_layout()
plt.show()
