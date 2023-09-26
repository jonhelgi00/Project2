import cv2
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import copy
import pickle



def Extract_features_database():

  mypath='/Users/jonhelgi/KTH_projects/Analysis_Search/Project2/Data2/server'
  onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
  images = []
  for n in range(0, len(onlyfiles)):
    images.append(cv2.imread(join(mypath,onlyfiles[n])))
  
  images[25] = copy.copy(images[24]) #images[25] is corrupt

  obj_features = {}
  des = []
  no_ft_img = 0
  for img_no in range(1,len(images)+1):
    gray = cv2.cvtColor(images[img_no-1], cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.12, edgeThreshold = 75) 
    kp, des_temp = sift.detectAndCompute(gray, None)
    # if images[img_no-1] is not None:
    #   print(img_no)
    # kp, des_temp = sift.detectAndCompute(images[img_no-1], None)
    des.append(des_temp)
    print(f"Number of keypoints in image: {len(kp)}")
    no_ft_img += len(kp)
    if np.mod(img_no,3) == 0 and img_no is not 0:
      obj_features[img_no/3] = des
      des = []

  avg_ft = no_ft_img/len(images)
  avg_ft_obj = no_ft_img/len(images)*3
  print(f"Number of features per image: {avg_ft}")
  print(f"Number of features per object: {avg_ft_obj}")
  return obj_features

database_ft = Extract_features_database()
with open('database_ft.pkl', 'wb') as fp:
  pickle.dump(database_ft, fp)
#13035 features per object
#4345 features per image

def Extract_features_query():

  mypath='/Users/jonhelgi/KTH_projects/Analysis_Search/Project2/Data2/client'
  onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
  images = []
  for n in range(0, len(onlyfiles)):
    images.append(cv2.imread(join(mypath,onlyfiles[n])))

  obj_features = {}
  des = []
  no_ft_img = 0
  sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.12, edgeThreshold = 75) 
  for img_no in range(1,len(images)+1):
    try:
      gray = cv2.cvtColor(images[img_no-1], cv2.COLOR_BGR2GRAY)
      kp, des = sift.detectAndCompute(gray, None)
      # if images[img_no-1] is not None:
      #   print(img_no)
      print(f"Number of keypoints in image: {len(kp)}")
      obj_features[img_no] = des
      des = []
    except:
      print(f"img_no: {img_no} is corrupt")  
    no_ft_img += len(kp)

  avg_ft = no_ft_img/len(images)
  print(f"Number of features per object: {avg_ft}")
  return obj_features      

query_ft = Extract_features_query() #img no. 7 is corrupt
with open('query_ft.pkl', 'wb') as fp:
  pickle.dump(query_ft, fp)
#4927 features per object    