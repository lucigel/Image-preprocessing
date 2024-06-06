import numpy as np
import cv2


def calculate_threshold(image):
    ma, na = image.shape
    mxn = ma * na
    g = []
    h = []
    t = []
    tong_i_hi = []
    m = []
    f = []

    anh_xuly = np.reshape(image, -1)
    anh_xuly = list(anh_xuly)

    g = sorted(list(set(anh_xuly)))
    len_g = len(g)

    dem = 0
    for i in range(len_g):
        tam = anh_xuly.count(g[i])
        h.append(tam)
        dem += tam
        t.append(dem)
    print(t)

    tam = 0
    for i in range(len_g):
        tam += g[i] * h[i]
        tong_i_hi.append(tam)

    for i in range(len_g):
        if t[i] == 0:
            m.append(float('inf'))
        else:
            m.append(1 / t[i] * tong_i_hi[i])

    for i in range(len_g):
        if mxn == t[i]:
            f.append(0)
        else:
            f.append(t[i] / (mxn - t[i]) * (m[i] - m[-1]) ** 2)
    O = 0
    f_max = max(f)
    for i in range(len_g):
        if f[i] == f_max:
            O = g[i]
            break

    anh_ra = np.copy(image)
    for x in range(ma):
        for y in range(na):
            if image[x][y] >= 0:
                anh_ra[x][y] = 255
            else:
                anh_ra[x][y] = 0
    print(O)
    return O


# Read the image
image = cv2.imread('anh2.png', cv2.IMREAD_GRAYSCALE)

# Calculate threshold
_, thresholded_image = cv2.threshold(image, calculate_threshold(image), 255, cv2.THRESH_BINARY)

cv2.imshow('Original Image', image)
cv2.imshow('Thresholded Image', thresholded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()