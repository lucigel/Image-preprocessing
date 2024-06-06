import time

import numpy as np
import cv2


def input_kernel():
    a = []
    t = 0
    for i in range(k):
        temp = []
        for j in range(k):
            x = int(input())
            temp.append(x)
            t += x
        a.append(temp)
    return a, t


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
            result[i][j] = int(
                np.round(np.sum(get_img_kernel(image, k, [i + num_padding, j + num_padding]) * kr) / sum_k))
    return result


k = int(input("Nhập k= "))
print("Nhập các phần tử của kernel: ")
kernel, sum_kernel = input_kernel()
print("Đã nhập xong______Đang tính toán")
img = cv2.imread("anh1_1.png", cv2.IMREAD_GRAYSCALE)
start = time.time()
img_tich_chap = tich_chap(img, kernel, sum_kernel, k // 2)
end = time.time()
print(end - start)
cv2.imshow("Anh ban dau", img)
cv2.imshow("Tich chap", img_tich_chap)

cv2.waitKey(0)
cv2.destroyAllWindows()