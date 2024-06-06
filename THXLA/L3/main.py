import cv2
import numpy as np

def load_image(filename):
    return cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

def show_image(image, window_name):
    cv2.imshow(window_name, image)


def dilation(image, kernel, iterations=1):
    dilated_image = np.copy(image)
    kernel = np.uint8(kernel)  # Convert kernel to uint8
    for _ in range(iterations):
        dilated_image = cv2.dilate(dilated_image, kernel)
    return dilated_image

def erosion(image, kernel, iterations=1):
    eroded_image = np.copy(image)
    kernel = np.uint8(kernel)  # Convert kernel to uint8
    for _ in range(iterations):
        eroded_image = cv2.erode(eroded_image, kernel)
    return eroded_image

if __name__ == "__main__":
    # Load image
    input_image = load_image("anh1.png")

    # Define kernel for dilation and erosion
    kernel = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ])



    # Apply erosion
    eroded_image = erosion(input_image, kernel, 7)
    # Apply dilation
    dilated_image = dilation(eroded_image, kernel, 7)

    dilated_image = dilation(dilated_image, kernel, 5)
    eroded_image = erosion(dilated_image, kernel, 5)



    # Show images
    show_image(input_image, "Input Image")
    show_image(dilated_image, "Result Image")

    cv2.waitKey(0)
    cv2.destroyAllWindows()