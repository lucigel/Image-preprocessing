import cv2
import numpy as np

image = cv2.imread('anh3.png', cv2.IMREAD_GRAYSCALE)


def otsu_value(img):
    height, width = img.shape

    histogram, _ = np.histogram(img, bins=256, range=(0, 256))

    total_pixels = height * width

    sum_total = np.sum(np.arange(256) * histogram)

    sum_back = 0

    num_back = 0

    threshold = -1

    variance_max = 0

    for i in range(256):
        num_back += histogram[i]
        if num_back == 0:
            continue

        num_fore = total_pixels - num_back
        if num_fore == 0:
            break

        sum_back += i * histogram[i]

        mean_back = sum_back / num_back
        mean_fore = (sum_total - sum_back) / num_fore

        variance_between = num_back * num_fore * (mean_back - mean_fore) ** 2

        if variance_between > variance_max:
            variance_max = variance_between
            threshold = i
    return threshold


_, thresholded_image = cv2.threshold(image, otsu_value(image), 255, cv2.THRESH_BINARY)
print(otsu_value(image))
cv2.imshow('Original Image', image)
cv2.imshow('Otsu Thresholded Image', thresholded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()