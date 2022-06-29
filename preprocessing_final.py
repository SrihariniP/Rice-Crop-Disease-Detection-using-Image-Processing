from math import copysign, log10
import numpy as np
from array import *
import os
import cv2
import csv
import sklearn
from skimage.feature import graycomatrix, graycoprops
from skimage import data
from PIL import Image

def resizeImage(img, target_w):
    img_h, img_w, d = img.shape
    image = img
    img_ratio = img_h/img_w
    target_h = target_w * img_ratio
    target_size = (round(target_w),round(target_h))
    #print(target_size)
    resized_img = cv2.resize(img, target_size, interpolation = cv2.INTER_AREA)
    #cv2.imshow("resized image", resized_img)
    return resized_img

def BackgroundElimination(imgMatrix):
    hsvImage = cv2.cvtColor(imgMatrix, cv2.COLOR_BGR2HSV)
    retval, thresh_gray = cv2.threshold(hsvImage, thresh=71, maxval=255, type=cv2.THRESH_BINARY)
    hsvMask = (thresh_gray[:, :, 1] > 250)  # & (thresh_gray[:,:,1]==0) & (thresh_gray[:,:,2]==0)
    imageNew = imgMatrix.copy()
    imageNew[:, :, 0] = imageNew[:, :, 0] * hsvMask
    imageNew[:, :, 1] = imageNew[:, :, 1] * hsvMask
    imageNew[:, :, 2] = imageNew[:, :, 2] * hsvMask
    return imageNew

def HealthyLeafElimination(imgMatrix):
    hsvImage1 = cv2.cvtColor(imgMatrix, cv2.COLOR_BGR2HSV)
    hueImage = hsvImage1[:, :, 1]
    retval1, thresh_gray1 = cv2.threshold(hsvImage1, thresh=22, maxval=255, type=cv2.THRESH_BINARY_INV)
    hsvMask1 = (thresh_gray1[:, :, 0] > 250)

    imageNew1 = imgMatrix.copy()
    imageNew1[:, :, 0] = imageNew1[:, :, 0] * hsvMask1
    imageNew1[:, :, 1] = imageNew1[:, :, 1] * hsvMask1
    imageNew1[:, :, 2] = imageNew1[:, :, 2] * hsvMask1

    return imageNew1

def cvt_infected_white(imgmatrix):
    hsv = cv2.cvtColor(imgmatrix, cv2.COLOR_BGR2HSV)
    retval1, thresh_gray1 = cv2.threshold(hsv, thresh=22, maxval=255, type=cv2.THRESH_BINARY_INV)
    Lower_hsv = np.array([250, 0, 0])
    Upper_hsv = np.array([255, 0, 0])

    Mask = cv2.inRange(thresh_gray1, Lower_hsv, Upper_hsv)

    return Mask

def cal_Hu_Moments(imgmatrix):
    moments = cv2.moments(imgmatrix)
    huMoments = cv2.HuMoments(moments)
    #print(huMoments)
    for i in range(0, 7):
        if(huMoments.all() != 0):
            huMoments[i] = -1 * copysign(1.0, huMoments[i]) * log10(abs(huMoments[i]))
    return huMoments

def getContourFeatures(img):

    contours, hierarchy = cv2.findContours(img, 1, 2)

    perimeter = 0
    area = 0
    num = 0
    #print(contours)
    for cnt in contours:
        p1 = cv2.arcLength(cnt, True)
        a1 = cv2.contourArea(cnt)

        #print (p1, a1)
        if(a1>0):
            perimeter += p1
            area += a1
            num += 1
    if(num != 0):
        avg_peri = perimeter/(num)
        avg_area = area/(num)
    #print(perimeter, area)
    else:
        avg_peri = 0
        avg_area = 0

    return avg_peri, avg_area

def GLCMfeatures(img, distance):
    gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    c_list = ["energy", "correlation", "dissimilarity", "homogeneity", "contrast"]
    avg = []
    count = 0
    for c in c_list:
        glcm_matrix_0 = graycomatrix(gs, [distance], [0])
        glcm_matrix_45 = graycomatrix(gs, [distance], [np.pi/4])
        glcm_matrix_90 = graycomatrix(gs, [distance], [np.pi/2])
        glcm_matrix_135 = graycomatrix(gs, [distance], [3*np.pi/4])
        deg0 = graycoprops(glcm_matrix_0, c)[0]
        deg45 = graycoprops(glcm_matrix_45, c)[0]
        deg90 = graycoprops(glcm_matrix_90, c)[0]
        deg135 = graycoprops(glcm_matrix_135, c)[0]

        #print(energy1)
        av = (deg0+deg45+deg90+deg135)/5
        #print(av)
        avg.append(av[0])
        count += 1
    return avg

def writeCSV(areas):
    f = open('E:/Sriharini/Rice_Crop_Disease_Detection/RCDD_dataset2.csv', 'w', newline ='', encoding='UTF8')

    with f:
        # create the csv writer
        writer = csv.writer(f)

        for i in areas:
            # write a row to the csv file
            writer.writerow(i)



data = []


head = ["BlueChannelMean", "GreenChannelMean", "RedChannelMean", "Bluestdev", "GreenStdev", "RedStdev",
       "AvgContourPerimeter", "AvgContourArea", "humoment1", "humoment2", "humoment3", "humoment4", "humoment5",
       "humoment6", "humoment7", "GLCM_energy_3", "GLCM_correlation_3", "GLCM_dissimilarity_3", "GLCM_homogeneity_3",
        "GLCM_contrast_3", "GLCM_energy_5", "GLCM_correlation_5", "GLCM_dissimilarity_5", "GLCM_homogeneity_5",
        "GLCM_contrast_5", "Outcome"]

data.append(head)
path = ('Rice_leaf_diseases')
#print("path = ",path)
for file in os.listdir(path):
    img_path = path + "/" + file
    print(img_path)
    for image in os.listdir(img_path):
        #Extracting Image
        img_path_1 = img_path + "/" + image
        img = cv2.imread(img_path_1)
        #original_img_name = "original" + " " + image
        #cv2.imshow(original_img_name, img)

        #Resizing Image
        resized_img = resizeImage(img, 512)
        #cv2.imshow("resized image", resized_img)

        #Background Elimination
        bge_img = BackgroundElimination(resized_img)
        #cv2.imshow("bge image", bge_img)

        #Elimination of healthy portion of leaf
        le_img = HealthyLeafElimination(bge_img)
        preprocess_img_name = "le image" + " " + image
        #cv2.imshow(preprocess_img_name, le_img)

        #Convertion of infected portion to white
        cvt_img = cvt_infected_white(resized_img)
        name = "cvt thresh grey" + " " + image
        #cv2.imshow(name, cvt_img)

        #Calculating Hu Moments
        huMoments = cal_Hu_Moments(cvt_img)
        #print("Hu Moments = ", huMoments)

        #get colours for infected area
        peri, area = getContourFeatures(cvt_img)

        #GLCM texture features
        t1 = GLCMfeatures(bge_img, 3)
        t2 = GLCMfeatures(bge_img, 5)

        #Mean of RGB channels
        #channels = cv2.mean(le_img)
        mean_channels, std_channels =cv2.meanStdDev(le_img, mask = None)
        #print("mean channels = ",mean_channels)
        lst = []
        for i in range(len(mean_channels)):
            lst.append(mean_channels[i][0])

        for i in range(len(std_channels)):
            lst.append(std_channels[i][0])

        lst.append(peri)
        lst.append(area)

        for i in range(len(huMoments)):
            lst.append(huMoments[i][0])

        #lst.append(t1)
        for i in range(len(t1)):
            lst.append(t1[i])

        for i in range(len(t2)):
            lst.append(t2[i])

        # Outcome is
        # 1 - Bacterial Leaf Blight
        # 2 - Brown Spot
        # 3 - Leaf Smut
        if(file == "Bacterial_leaf_blight"):
            lst.append(1)

        elif(file == "Brown_spot"):
            lst.append(2)

        else:
            lst.append(3)

        #print("lst = ", lst)
        data.append(lst)

writeCSV(data)

cv2.waitKey(0)
cv2.destroyAllWindows()