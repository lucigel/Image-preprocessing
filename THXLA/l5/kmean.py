import cv2
import numpy as np
from sklearn.cluster import KMeans

image = cv2.imread('img.png')

image_flattened = image.reshape((-1, 3)) # Chuyển ảnh thành mảng 2D

kmeans = KMeans(n_clusters=20, random_state=0)

kmeans.fit(image_flattened)

centers = kmeans.cluster_centers_

segmented_image = np.zeros_like(image)
labels = kmeans.labels_
for i in range(len(labels)):
    row_index = i // image.shape[1]
    col_index = i % image.shape[1]

    label = labels[i]

    # Lấy giá trị trung tâm tương ứng với nhãn của pixel từ mảng centers
    center_color = centers[label]

    # Gán giá trị màu của trung tâm vào vị trí tương ứng trong ảnh phân cụm
    segmented_image[row_index, col_index] = center_color

# Hiển thị ảnh gốc và ảnh đã phân cụm
cv2.imshow('Original Image', image)
cv2.imshow('Segmented Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()