import copy
import numpy as np
import cv2
import csv
imgId = ['1.PNG', '2.PNG','3.PNG','4.PNG','5.PNG']
values = []
for img in imgId:
    outputImage = cv2.imread('D:\mipt\\5-2\image analysis\Dataset\output\modified otsu\\halftone\\'+img, 0)
    mask = cv2.imread('D:\mipt\\5-2\image analysis\\Dataset\\new masks\\'+img, 0)

    TP = 0
    TN = 0
    FP = 0
    FN = 0
    w, h = outputImage.shape
    for i in range(w):
        for j in range(h):
            if (mask[i, j] == 0):
                if (outputImage[i, j] == 0):
                    TP += 1
                else:
                    FP += 1
            if (mask[i, j] == 255):
                if (outputImage[i, j] == 255):
                    TN += 1
                else:
                    FN += 1

    # calculating the mean square error MSE
    maskNew = copy.deepcopy(mask)
    outImageNew = copy.deepcopy(outputImage)
    maskNew = maskNew.astype('float64')
    outImageNew = outImageNew.astype('float64')
    maskNew = maskNew / 255
    outImageNew = outImageNew / 255
    numerator = 0
    denominator = 0
    for i in range(w):
        for j in range(h):
            if (maskNew[i, j] == 0 or maskNew[i, j] == 1):
                temp = (maskNew[i, j] - outImageNew[i, j]) ** 2
                numerator += temp
                denominator += 1
    MSE = numerator / denominator

    # calculating Recall, Precision, and F1
    Recall = TP / (TP + FN)
    Precision = TP / (TP + FP)
    F1 = 2 * Recall * Precision / (Recall + Precision)
    print('Recall = ', np.round(Recall, 4))
    print('Precision = ', np.round(Precision, 4))
    print('F1 = ', np.round(F1, 4))
    print('MSE = ', np.round(MSE, 4))
    valuesOfOneImage = [np.round(Recall, 4), np.round(Precision, 4), np.round(F1, 4), np.round(MSE, 4)]
    values.append(valuesOfOneImage)
    print(valuesOfOneImage)
    # cv2.imshow('out', outputImage)
    # cv2.imshow('mask', mask)

    # cv2.waitKey(0)

with open('D:\mipt\\5-2\image analysis\\data.csv', 'w', newline='') as csvfile:
    r = csv.writer(csvfile, delimiter=',')
    for row in values:
        r.writerow(row)