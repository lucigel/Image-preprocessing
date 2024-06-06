import cv2
import numpy as np

# Đọc ảnh đen trắng
img = cv2.imread('anh3.png', cv2.IMREAD_GRAYSCALE)


def sobel(image):
    height, width = image.shape

    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])

    gradient_x = np.zeros((height, width), dtype=np.float32)
    gradient_y = np.zeros((height, width), dtype=np.float32)

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            gx = np.sum(image[i - 1:i + 2, j - 1:j + 2] * sobel_x)
            gy = np.sum(image[i - 1:i + 2, j - 1:j + 2] * sobel_y)
            gradient_x[i, j] = gx
            gradient_y[i, j] = gy

    return gradient_x, gradient_y


h, w = img.shape
x, y = sobel(img)

gradient_magnitude = np.sqrt(x ** 2 + y ** 2)
gradient_direction = np.arctan2(y, x)

thresholded_magnitude = np.zeros((h, w), dtype=np.uint8)
thresholded_magnitude[gradient_magnitude > 100] = 255

# Hiển thị ảnh gốc và ảnh nhị phân của biên
cv2.imshow('Original Image', img)
cv2.imshow('Edge Detected Image', thresholded_magnitude)
cv2.waitKey(0)
cv2.destroyAllWindows()