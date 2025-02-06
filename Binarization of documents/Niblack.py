import cv2
import numpy as np
import copy

inputImage = cv2.imread('D:\mipt\\5-2\image analysis\Dataset\input\\5.jpg', 0)

wx = 11
wy = 11
x, y = [2 * wx + 1, 2 * wy + 1]
sizeOfKernal = x * y
k = -0.2
a = 0
outputImage = copy.deepcopy(inputImage)

kernel = np.ones((x, y), np.float32) / sizeOfKernal
meanOfImage = cv2.filter2D(inputImage, -1, kernel)
squaresOfPixels = np.square(inputImage.astype('float64'))
meanOfSquares = cv2.filter2D(squaresOfPixels, -1, kernel)
deviationOfImage = cv2.sqrt(meanOfSquares - meanOfImage)

for i in range(inputImage.shape[0]):
    for j in range(inputImage.shape[1]):
        localMean = meanOfImage[i, j]
        localDeviation = deviationOfImage[i, j]
        localThreshold = localMean + k * localDeviation + a
        if inputImage[i, j] < localThreshold:
            outputImage[i, j] = 0
        else:
            outputImage[i, j] = 255

cv2.imshow('output', outputImage)
cv2.waitKey(0)
