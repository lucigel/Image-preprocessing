import numpy as np
import cv2

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
# Hàm lấy vị trí cần nhân tích chập trong mảng
def get_img_kernel(image, kernel_size, center):
    padding_size = kernel_size // 2  # Kích thước padding cần thêm
    padded_image = np.pad(image, ((padding_size, padding_size), (padding_size, padding_size)), mode='constant')
    start_row = center[0] - kernel_size // 2
    start_col = center[1] - kernel_size // 2
    end_row = start_row + kernel_size
    end_col = start_col + kernel_size
    img_result = padded_image[start_row:end_row, start_col:end_col]
    return img_result


def tich_chap(image, kr, sum_k, num_padding):
    result = np.copy(img).astype(np.uint8)
    img_height, img_width = image.shape
    for i in range(img_height):
        for j in range(img_width):
            temp = int(np.abs(np.round(np.sum(get_img_kernel(image, k, [i + num_padding, j + num_padding]) * kr) / sum_k)))
            result[i][j] = temp

    print(otsu_value(result))
    _, thresholded_image = cv2.threshold(result, 27, 255, cv2.THRESH_BINARY)
    return thresholded_image


k = 3
kernel1 = np.array([[0, -1, 0],
                    [-1, 4, -1],
                    [0, -1, 0]])
kernel2 = np.array([[-1, -1, -1],
                    [-1, 8, 1],
                    [-1, -1, -1]])
kernel3 = np.array([[1, -2, 1],
                    [-2, 4, -2],
                    [1, -2, 1]])

print("Đã nhập xong______Đang tính toán")
img = cv2.imread("anh2.png", cv2.IMREAD_GRAYSCALE)
img_laplace1 = tich_chap(img, kernel1, 1, k // 2)
img_laplace2 = tich_chap(img, kernel2, 1, k // 2)
img_laplace3 = tich_chap(img, kernel3, 1, k // 2)

cv2.imshow("Anh ban dau", img)
cv2.imshow("Laplace_1", img_laplace1)
cv2.imshow("Laplace_2", img_laplace2)
cv2.imshow("Laplace_3", img_laplace3)

cv2.waitKey(0)
cv2.destroyAllWindows()