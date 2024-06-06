import cv2
import numpy as np

# Define your homogeneity criterion function here
def homogeneity_criterion(region):
    # Example: Check if standard deviation is smaller than a threshold
    threshold = 1
    return np.std(region) <= threshold

# Define a Region class to represent each region
class Region:
    def __init__(self, image, roi):
        self.image = image
        self.roi = roi
        self.validity = True
        self.children = []
        self.label = None
        self.mask = None  # Initialize the mask attribute

# Function to merge two adjacent regions
def merge_two_regions(src, r1, r2, predicate):
    roi1 = r1.roi
    roi2 = r2.roi

    # Calculate the bounding rectangle that encompasses both regions
    x = min(roi1[0], roi2[0])
    y = min(roi1[1], roi2[1])
    w = max(roi1[0] + roi1[2], roi2[0] + roi2[2]) - x
    h = max(roi1[1] + roi1[3], roi2[1] + roi2[3]) - y

    roi12 = (x, y, w, h)

    region1 = src[roi1[1]:roi1[1]+roi1[3], roi1[0]:roi1[0]+roi1[2]]
    region2 = src[roi2[1]:roi2[1]+roi2[3], roi2[0]:roi2[0]+roi2[2]]
    if predicate(region1) and predicate(region2):
        r1.roi = roi12
        r1.mask = np.zeros_like(src, dtype=np.uint8)
        cv2.rectangle(r1.mask, (roi12[0], roi12[1]), (roi12[0]+roi12[2], roi12[1]+roi12[3]), 1, cv2.FILLED)
        r2.validity = False
        return True
    return False


# Function to merge adjacent regions recursively
def merge(src, r, predicate):
    if len(r.children) < 1:
        return
    row1 = merge_two_regions(src, r.children[0], r.children[1], predicate)
    row2 = merge_two_regions(src, r.children[2], r.children[3], predicate)
    if not (row1 or row2):
        col1 = merge_two_regions(src, r.children[0], r.children[2], predicate)
        col2 = merge_two_regions(src, r.children[1], r.children[3], predicate)
    for child in r.children:
        if len(child.children) > 0:
            merge(src, child, predicate)

# Function to split a region into four quadrants
def split(src, roi, predicate):
    r = Region(src, roi)
    if predicate(src):
        mean = np.mean(src)
        r.label = mean
    else:
        h, w = src.shape
        h_mid = h // 2
        w_mid = w // 2
        r.children.append(split(src[:h_mid, :w_mid], (roi[0], roi[1], h_mid, w_mid), predicate))
        r.children.append(split(src[:h_mid, w_mid:], (roi[0] + w_mid, roi[1], h_mid, w - w_mid), predicate))
        r.children.append(split(src[h_mid:, :w_mid], (roi[0], roi[1] + h_mid, h - h_mid, w_mid), predicate))
        r.children.append(split(src[h_mid:, w_mid:], (roi[0] + w_mid, roi[1] + h_mid, h - h_mid, w - w_mid), predicate))
    return r

# Function to traverse and print the regions
def print_regions(r):
    if r.validity and not r.children:
        print(r.mask, "at", r.roi)
    for child in r.children:
        print_regions(child)

# Function to draw rectangles of regions on an image
def draw_rect(imgRect, r):
    if r.validity and not r.children:
        cv2.rectangle(imgRect, (r.roi[0], r.roi[1]), (r.roi[0]+r.roi[2], r.roi[1]+r.roi[3]), 50, -1)
    for child in r.children:
        draw_rect(imgRect, child)

# Function to draw filled rectangles of regions on an image
def draw_region(img, r):
    if r.validity and not r.children:
        cv2.rectangle(img, (r.roi[0], r.roi[1]), (r.roi[0]+r.roi[2], r.roi[1]+r.roi[3]), r.label, cv2.FILLED)
    for child in r.children:
        draw_region(img, child)

# Split&merge test predicates
def predicate_std_zero(src):
    stddev = np.std(src)
    return stddev == 0

def predicate_std_5(src):
    stddev = np.std(src)
    return stddev <= 5.8 or src.size <= 25

img = cv2.imread("anh3.png", 0)

cv2.imshow("original", img)

print("now try to split..")
r = split(img, (0, 0, img.shape[1], img.shape[0]), predicate_std_5)

print("splitted")
imgRect = img.copy()
draw_rect(imgRect, r)

merge(img, r, predicate_std_5)
imgMerge = img.copy()
draw_rect(imgMerge, r)
imgSegmented = img.copy()

draw_region(imgSegmented, r)
cv2.imshow("segmented", imgSegmented)
cv2.imwrite("segmented.jpg", imgSegmented)

cv2.waitKey(0)
cv2.destroyAllWindows()