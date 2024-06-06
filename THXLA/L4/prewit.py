import cv2
import numpy as np

img = cv2.imread('anh1.png', cv2.IMREAD_GRAYSCALE)

def prewitt(image):
    height, width = image.shape

    prewitt_x = np.array([[-1, 0, 1],
                          [-1, 0, 1],
                          [-1, 0, 1]])

    prewitt_y = np.array([[-1, -1, -1],
                          [0, 0, 0],
                          [1, 1, 1]])

    gradient_x = np.zeros((height, width), dtype=np.float32)
    gradient_y = np.zeros((height, width), dtype=np.float32)

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            gx = np.sum(image[i - 1:i + 2, j - 1:j + 2] * prewitt_x)
            gy = np.sum(image[i - 1:i + 2, j - 1:j + 2] * prewitt_y)
            gradient_x[i, j] = gx
            gradient_y[i, j] = gy
    return gradient_x, gradient_y


x, y = prewitt(img)
h, w = img.shape

gradient_magnitude = np.sqrt(x ** 2 + y ** 2)
gradient_direction = np.arctan2(y, x)

thresholded_magnitude = np.zeros((h, w), dtype=np.uint8)
thresholded_magnitude[gradient_magnitude > 100] = 255

cv2.imshow('Original Image', img)
cv2.imshow('Edge Detected Image', thresholded_magnitude)
cv2.waitKey(0)
cv2.destroyAllWindows()