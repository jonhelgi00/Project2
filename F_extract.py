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
  imgDict = {}
  for name in onlyfiles:
    try:
      imgDict[name]= cv2.imread(join(mypath,name))
    except:
      print(f"Name: {name} does not work")
  
  # images[25] = copy.copy(images[24]) #images[25] is corrupt
  obj_features = {}
  des = []
  obj_no = None
  kpNoDict = {}
  sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.13, edgeThreshold = 75) 

  for fname in imgDict:
    string = fname[3:5]

    try:
      if "_" in string:
        obj_no = int(string[0])
      else:
        obj_no = int(string)
      print(obj_no)
    except:
      print(f"fname obj_no extraction not working")   

    try:
      gray = cv2.cvtColor(imgDict[fname], cv2.COLOR_BGR2GRAY)
      kp, des_temp = sift.detectAndCompute(gray, None)
      print(f"Number of keypoints in image: {len(kp)}")
      if obj_no in obj_features:
        obj_features[obj_no].append(des_temp)
      else:
        obj_features[obj_no] = des_temp

      if obj_no in kpNoDict:
        kpNoDict[obj_no] += len(kp)
      else:
        kpNoDict[obj_no] = len(kp)
    
    except:
      print(f"fname {fname} kp extraction not working")

  avg_obj = 0
  avg_img = 0
  for kpNo in kpNoDict.values():
    avg_obj += kpNo

  avg_obj /= 50
  avg_img = avg_obj / 3

  return obj_features, avg_obj, avg_img


database_ft, avg_obj, avg_img = Extract_features_database()
with open('database_ft_new.pkl', 'wb') as fp:
  pickle.dump(database_ft, fp)
print(f"avg_obj: {avg_obj}")
print(f"avg_img: {avg_img}")
#3336 features per object
#1111 features per image

# def Extract_features_query():

#   mypath='/Users/jonhelgi/KTH_projects/Analysis_Search/Project2/Data2/client'
#   onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
#   images = []
#   for n in range(0, len(onlyfiles)):
#     images.append(cv2.imread(join(mypath,onlyfiles[n])))

#   obj_features = {}
#   des = []
#   no_ft_img = 0
#   sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.12, edgeThreshold = 75) 
#   for img_no in range(1,len(images)+1):
#     try:
#       gray = cv2.cvtColor(images[img_no-1], cv2.COLOR_BGR2GRAY)
#       kp, des = sift.detectAndCompute(gray, None)
#       # if images[img_no-1] is not None:
#       #   print(img_no)
#       print(f"Number of keypoints in image: {len(kp)}")
#       obj_features[img_no] = des
#       des = []
#     except:
#       print(f"img_no: {img_no} is corrupt")  
#     no_ft_img += len(kp)

#   avg_ft = no_ft_img/len(images)
#   print(f"Number of features per object: {avg_ft}")
#   return obj_features      

# query_ft = Extract_features_query() #img no. 7 is corrupt
# with open('query_ft.pkl', 'wb') as fp:
#   pickle.dump(query_ft, fp)
# #4927 features per object    