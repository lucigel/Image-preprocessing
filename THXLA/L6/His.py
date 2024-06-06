import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors
import os


def calculate_histogram(image, space='rgb'):
    if space == 'rgb':
        hist_bins = [26, 26, 26]  # Số lượng bin cho mỗi kênh màu RGB
    elif space == 'hsv':
        hist_bins = [24, 5, 5]  # Số lượng bin cho mỗi kênh màu HSV

    hist = np.zeros(hist_bins)

    if space == 'hsv':
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        for i in range(hsv_image.shape[0]):
            for j in range(hsv_image.shape[1]):
                h = int(hsv_image[i, j, 0] / 15)  # Chia bin cho Hue
                s = int(hsv_image[i, j, 1] * 5)  # Chia bin cho Saturation
                if s >= hist.shape[1]:
                    s = hist.shape[1] - 1
                v = int(hsv_image[i, j, 2] * 5)  # Chia bin cho Value
                if v >= hist.shape[2]:
                    v = hist.shape[2] - 1
                hist[h, s, v] += 1
    else:
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                b = int(image[i, j, 0] / 10)  # Chia bin cho Blue
                g = int(image[i, j, 1] / 10)  # Chia bin cho Green
                r = int(image[i, j, 2] / 10)  # Chia bin cho Red
                hist[r, g, b] += 1

    hist /= np.sum(hist)
    return hist.flatten()


def find_similar_images(test_image, data_images, k=3, space='rgb'):
    test_hist = calculate_histogram(test_image, space=space)
    data_hist = [calculate_histogram(img, space=space) for img in data_images]
    data_hist = np.array(data_hist).reshape(len(data_hist), -1)

    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(data_hist)
    distances, indices = knn.kneighbors([test_hist.flatten()])

    return indices[0]


def read_images_from_directory(directory):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            img_path = os.path.join(directory, filename)
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
    return images


# Load ảnh Test và Data
test_image = cv2.imread("test/a10.jpg")
data_directory = "data"
data_images = read_images_from_directory(data_directory)

# Tìm K ảnh Data gần nhất với ảnh Test
similar_indices = find_similar_images(test_image, data_images, space='hsv')

# Hiển thị ảnh Test và K ảnh Data
cv2.imshow("Test Image", test_image)
for idx in similar_indices:
    cv2.imshow(f"Similar Image {idx}", data_images[idx])

cv2.waitKey(0)
cv2.destroyAllWindows()
