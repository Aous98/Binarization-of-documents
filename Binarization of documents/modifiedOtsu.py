import cv2
import math


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


def getVariance(k, ub, uf, his):
    sumHis1 = 0
    sumHis2 = 0
    sum1 = 0
    sum2 = 0
    varianceB = 1000
    varianceF = 1000
    for i in range(k):
        sumHis1 += his[i]
        sum1 += his[i] * (i - ub) ** 2
    if sumHis1 != 0:
        varianceB = sum1 / sumHis1
    for i in range(k, 256):
        sumHis2 += his[i]
        sum2 += his[i] * (i - uf) ** 2
    if sumHis2 != 0:
        varianceF = sum2 / sumHis2

    return varianceB, varianceF


def getThresholdValue(his):
    k = 0
    Q_best = -1000000000
    Q = -100000000
    for i in range(1, 256):
        wb, wf, ub, uf = getPropability(i, his)
        segmaB, segmaF = getVariance(i, ub, uf, his)
        segma = wb * segmaB + wf * segmaF
        if (wb != 0 and wf != 0):
            Q = wb * math.log(wb) + wf * math.log(wf) - 2 * math.log(segma)
        if Q > Q_best:
            k = i
            Q_best = Q

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
