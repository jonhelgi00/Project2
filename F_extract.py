import cv2
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import copy



def Extract_features():

  mypath='/Users/jonhelgi/KTH_projects/Analysis_Search/Project2/Data2/server'
  onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
  images = []
  for n in range(0, len(onlyfiles)):
    images.append(cv2.imread(join(mypath,onlyfiles[n])))
  # cv2.imshow("matches", images[26])
  # cv2.waitKey(0)
  # cv2.destroyAllWindows() 
  images[25] = copy.copy(images[24]) #images[25] is corrupt

  obj_features = {}
  des = []
  for img_no in range(1,len(images)+1):
    # gray = cv2.cvtColor(images[img_no-1], cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.15, edgeThreshold = 80) 
    # kp, des_temp = sift.detectAndCompute(gray, None)
    if images[img_no-1] is not None:
      print(img_no)
    kp, des_temp = sift.detectAndCompute(images[img_no-1], None)
    des.append(des_temp)
    print(f"Number of keypoints in image: {len(kp)}")
    if np.mod(img_no,3) == 0 and img_no is not 0:
      obj_features[img_no/3] = des
      des = []


Extract_features()

      

  