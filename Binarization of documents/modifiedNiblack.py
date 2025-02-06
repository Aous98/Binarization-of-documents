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
img = copy.deepcopy(inputImage)
img = img.astype('float64')
kernel = np.ones((x, y), np.float32) / sizeOfKernal
meanOfImage = cv2.filter2D(inputImage, -1, kernel)
squaresOfPixels = np.square(inputImage.astype('float64'))
meanOfSquares = cv2.filter2D(squaresOfPixels, -1, kernel)
deviationOfImage = cv2.sqrt(meanOfSquares - meanOfImage)

# for the douple window
newX, newY = [4 * wx + 1, 4 * wy + 1]
sizeOfKernal = newX * newY
newKernel = np.ones((newX, newY), np.float32) / sizeOfKernal
newMeanOfImage = cv2.filter2D(inputImage, -1, newKernel)
newSquaresOfPixels = np.square(inputImage.astype('float64'))
newMeanOfSquares = cv2.filter2D(newSquaresOfPixels, -1, newKernel)
newDeviationOfImage = cv2.sqrt(newMeanOfSquares - newMeanOfImage)


difference = (img - meanOfImage) ** 2

varianceOfImage = cv2.filter2D(difference, -1, kernel)
# print(varianceOfImage)
o = 0
for i in range(inputImage.shape[0]):
    for j in range(inputImage.shape[1]):
        localMean = meanOfImage[i, j]
        localDeviation = deviationOfImage[i, j]
        localThreshold = localMean + k * localDeviation + a

        if varianceOfImage[i, j] < localThreshold:
            newLocalMean = newMeanOfImage[i, j]
            newLocalDeviation = newDeviationOfImage[i, j]
            localThreshold = newLocalMean + k * newLocalDeviation + a
            if inputImage[i, j] < localThreshold:
                outputImage[i, j] = 0
            else:
                outputImage[i, j] = 255
        else:
            if inputImage[i, j] < localThreshold:
                outputImage[i, j] = 0
            else:
                outputImage[i, j] = 255
print(o)
cv2.imshow('output', outputImage)
cv2.waitKey(0)
