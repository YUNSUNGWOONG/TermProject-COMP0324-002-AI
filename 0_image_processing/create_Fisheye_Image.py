import cv2
import os
from PIL import Image
import numpy as np

def distort_image(img):
    height, width = img.shape[:2]
    # Camera Intrinsic Matrix
    K = np.array([[width / 2, 0, width / 2], # x축 초점 거리 / 0 / 주점 x 좌표
                  [0, width / 2, height / 2], # 0 / y축 초점 거리 / 주점 y좌표
                  [0, 0, 1]], dtype=np.float32) # 0 / 0 / 동차 좌표
    
    D = np.array([-0.9, 0.3, 1.5, 1.5], dtype=np.float32) # 왜곡 계수 // k1, k2: 방사형 왜곡 계수 / k3, k4: 접선 왜곡 계수
    
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, (width, height), cv2.CV_16SC2)
    distorted = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
    return distorted

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")
        
def resizeImg(originPath, name, destDir):
    origin = Image.open(originPath + name)
    width, height = origin.size
    imgSize = min(width, height)
    
    left = (width - imgSize) // 2
    top = (height - imgSize) // 2
    right = left + imgSize
    bottom = top + imgSize
    
    img_cropped = origin.crop((left, top, right, bottom))
    createDirectory(destDir)
    img_cropped.save(destDir + name)
    
for i in ["road/", "residential/", "city/", "campus/"]:
    count = 0
    path = "./dataset/0_rawData/"
    destPath = "./dataset/distortedData/"
    list = os.listdir(path + i)
    list = [file for file in list if (file.endswith(".png") or file.endswith(".jpg"))]
    print("Processing in " + path + i + "...")
    for j in list:
        resizeImg(path + i, j, "./dataset/resizedData/" + i)
        img = cv2.imread("./dataset/resizedData/" + i + j)
        fisheye_img = distort_image(img)
        createDirectory(destPath + i)
        cv2.imwrite(destPath + i + j, fisheye_img)
        count = count + 1
        print(str(count) + "/" + str(len(list)) + "... (" + destPath + i + j + ")")
    print("Finished!\n\n")


