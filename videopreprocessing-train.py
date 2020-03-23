import av
import glob
import os
import time
import tqdm
import datetime
import argparse
import cv2
import numpy as np

def video_to_frame(path,out_path):
    vidcap = cv2.VideoCapture(path)
    success,image = vidcap.read()
    count = 0
    while success:
      cv2.imwrite(os.path.join(out_path,"{}.jpg".format(count)), image)
      success,image = vidcap.read()
      count += 1
      
def extract_frames(video_path):
    frames = []
    video = av.open(video_path)
    for frame in video.decode(0):
        yield frame.to_image()
        

##Part-1
path = 'E:\\BE-PROJECT\\crimeucfdataset\\Anomaly_Dataset\\Anomaly_Videos\\Anomaly-Videos-Part-1'
result = 'E:\\BE-PROJECT\\Dataset'
from tqdm.autonotebook import tqdm
for i in tqdm(os.listdir(path)):
  p1 = os.path.join(path,i)
  r1 = os.path.join(result,i)
  if os.path.exists(r1):
            continue
  os.makedirs(r1,exist_ok = True)
  for j in os.listdir(p1):
    vid_path = os.path.join(p1,j)
    r2 = os.path.join(r1,j[:-4])
    os.makedirs(r2,exist_ok = True)
    for j, frame in enumerate((extract_frames(vid_path))):
      frame.save(os.path.join(r2, f"{j}.jpg"))        
      



##Part-2
from tqdm.autonotebook import tqdm
path = 'E:\\BE-PROJECT\\crimeucfdataset\\Anomaly_Dataset\\Anomaly_Videos\\Anomaly-Videos-Part-2'
result = 'E:\\BE-PROJECT\\Dataset'

for i in tqdm(os.listdir(path)):
  p1 = os.path.join(path,i)
  r1 = os.path.join(result,i)
  if os.path.exists(r1):
            continue
  os.makedirs(r1,exist_ok = True)
  for j in os.listdir(p1):
    vid_path = os.path.join(p1,j)
    r2 = os.path.join(r1,j[:-4])
    os.makedirs(r2,exist_ok = True)
    for j, frame in enumerate((extract_frames(vid_path))):
      frame.save(os.path.join(r2, f"{j}.jpg"))      
      

##Normal video      
from tqdm.autonotebook import tqdm
path = 'E:\\BE-PROJECT\\crimeucfdataset\\Anomaly_Dataset\\Anomaly_Videos\\Normal-Videos-Part-1'
result = 'E:\\BE-PROJECT\\Dataset\\normal'

for i in tqdm(os.listdir(path)):
  p1 = os.path.join(path,i)
  r1 = os.path.join(result,i[:-4])
  if os.path.exists(r1):
            continue
  os.makedirs(r1,exist_ok = True)
  for k, frame in enumerate((extract_frames(p1))):
    frame.save(os.path.join(r1, f"{k}.jpg"))      
    
 
    
    


path = 'E:\\BE-PROJECT\\Dataset'
res = 'E:\\BE-PROJECT\\crime16'
#Number
seq_length = 16

def preprocess_data(seq_length,path,res):
  dir = os.listdir(path)
  for i in tqdm(dir):
      p1 = os.path.join(path,i)
      r1 = os.path.join(res,i)
      os.makedirs(r1,exist_ok = True)
      for j in os.listdir(p1):
          p2 = os.path.join(p1,j)
          r2 = os.path.join(r1,j)
          l = 0
          skip_length = int(len(os.listdir(p2))/seq_length)
          for m in range(10):
              k = m
              while(l!=seq_length):

                  p3 = os.path.join(p2,str(k) + ".jpg")
                  try:
                      img = cv2.imread(p3)
                      img = cv2.resize(img,(128,128))
                  except:
                      print(p3)
                  if(k==0):
                      img1 = img
                  else:
                      img1 = np.append(img1,img,axis = 1)
                  k = k+skip_length
                  l = l+1    
              cv2.imwrite(r2 + str(m)+".jpg",img1)
              
preprocess_data(seq_length,path,res)              