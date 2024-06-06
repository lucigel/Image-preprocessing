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
    padding_size = kernel_size // 2  # Kích thước padding cần thêm (thêm padding để tính các điểm ở dìa)
    padded_image = np.pad(image, ((padding_size, padding_size),
                                  (padding_size, padding_size)), mode='constant')
    start_row = center[0] - kernel_size // 2
    start_col = center[1] - kernel_size // 2
    end_row = start_row + kernel_size
    end_col = start_col + kernel_size
    img_result = padded_image[start_row:end_row, start_col:end_col]

    median_value = np.median(img_result)
    print(median_value)
    return np.sum(img_result) / (kernel_size * kernel_size), median_value

def trung_vi(image, nguong, num_padding):
    result = np.copy(image).astype(np.uint8)
    img_height, img_width = image.shape
    for i in range(img_height):
        for j in range(img_width):
            avg, med = get_img_kernel(image, k, [i + num_padding, j + num_padding])
            temp = np.abs(med - result[i][j])
            if temp > nguong:
                result[i][j] = med

    result = np.array(result)
    return result


k = int(input("Nhập k= "))

print("Đã nhập xong______Đang tính toán")

img = cv2.imread("anh3_1.png", cv2.IMREAD_GRAYSCALE)
img = np.array(img)
img_trung_vi = trung_vi(img, 10, k // 2)
count = 0
comparison = img == img_trung_vi

print(comparison)
cv2.imshow("Anh ban dau", img)
cv2.imshow("Trung vi", img_trung_vi)

cv2.waitKey(0)
cv2.destroyAllWindows()