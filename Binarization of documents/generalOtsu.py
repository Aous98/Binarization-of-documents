import cv2


def getHistogram(img):
    w, h = img.shape
    r = 0
    his = [0 for i in range(256)]
    for i in range(w):
        for j in range(h):
            his[img[i][j]] += 1
            r += 1
    return his


def getPropability(k, his):
    global numberOfPixels
    w0 = 0
    w1 = 0
    u0 = 0
    u1 = 0
    for i in range(k):
        w0 += his[i]
        u0 += his[i] * i
    if w0 != 0:
        u0 = u0 / w0
    w0 = w0 / numberOfPixels
    for i in range(k, 256):
        w1 += his[i]
        u1 += his[i] * i
    if w1 != 0:
        u1 = u1 / w1
    w1 = w1 / numberOfPixels
    return w0, w1, u0, u1


def getThresholdValue(his):
    k = 0
    segmaBest = 0
    for i in range(1, 256):
        wb, wf, ub, uf = getPropability(i, his)
        segma = wb * wf * (ub - uf) ** 2
        if segma > segmaBest:
            k = i
            segmaBest = segma
    return k



inputImage = cv2.imread('input image direction', 0)  # input image in grayscale
cv2.imshow('input', inputImage)

W, H = inputImage.shape
numberOfPixels = W * H
histogram = getHistogram(inputImage)
thresholdValue = getThresholdValue(histogram)
print(thresholdValue)
inputImage[inputImage < thresholdValue] = 0
inputImage[inputImage >= thresholdValue] = 255

cv2.imshow('output image', inputImage)
cv2.waitKey(0)
