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
    print(img_result)

    return np.sum(img_result) / (kernel_size * kernel_size)


def trung_binh(image, nguong, num_padding):
    result = np.copy(image).astype(np.uint8)
    img_height, img_width = image.shape
    for i in range(img_height):
        for j in range(img_width):
            temp = np.abs(result[i][j] - get_img_kernel(image, k, [i + num_padding, j + num_padding]))
            if temp > nguong:
                result[i][j] = get_img_kernel(image, k, [i + num_padding, j + num_padding])
    return result


k = int(input("Nhập k= "))

print("Đã nhập xong______Đang tính toán")
img = cv2.imread("anh2_2.png", cv2.IMREAD_GRAYSCALE)
img_tich_chap = trung_binh(img, 30, k // 2)

cv2.imshow("Anh ban dau", img)
cv2.imshow("Trung binh", img_tich_chap)

cv2.waitKey(0)
cv2.destroyAllWindows()