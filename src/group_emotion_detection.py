#!/usr/bin/python3
#-*- coding: UTF-8 -*-

import roslib
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import os
import imutils
import numpy as np
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import tensorflow as tf


# Define function to visualize predictions
def visualize_predictions(a):
  pixels = a.astype('float32')
  samples = expand_dims(pixels, axis=0)
  samples = preprocess_input(samples, version=1)
  clases=custom_vgg_model.predict(samples)
  c=np.amax(clases)
  d=np.where(clases == c)
  m=int(d[1])
  if m == 0:
    n='Enojado'
  elif m == 1:
    n='Disgusto'
  elif m == 2:
    n='Miedo'
  elif m == 3:
    n='Feliz'
  elif m == 4:
    n='Triste'
  elif m == 5:
    n='Sorpresa'
  return m,n


def escenas(i,fr,area2,armax,lista, lista1,listo,listo1,f,g,esce):
  if len(area2)==fr:
    armaxi=max(area2[i:fr])
    armax.append(armaxi)
    fr=fr+20
    i=i+20
    if len(armax)>1:
      if armax[f-1]>armax[f]:
        mayor=armax[f]
        lista.append(f-1)
        listo.append(mayor)
        if len(lista)==1:
          lista1.append(lista[0])
          listo1.append(listo[0])
          np=listo[0]
          esce=esce+1
        if len(lista)>1:
          if abs(lista[g-1]-lista[g])!=1:
            np=listo[g]
            lista1.append(lista[g])
            listo1.append(listo[g])
            esce=esce+1
          g=g+1
      f=f+1
  return esce,i,fr,f,g

def emocion_frame(khe,khe2,emo,frame1, preds):
    if khe == khe2:
        emo.append(int(np.where(preds==np.amax(preds))[0]))
    c0=0
    c1=0
    c2=0
    c3=0
    c4=0
    c5=0
    for i in range(len(emo)):
        if emo[i] == 0:
            c0=c0+1
        if emo[i] == 1:
            c1=c1+1
        if emo[i] == 2:
            c2=c2+1
        if emo[i] == 3:
            c3=c3+1
        if emo[i] == 4:
            c4=c4+1
        if emo[i] == 5:
            c5=c5+1
    cant=[c0,c1,c2,c3,c4,c5]
    cant=np.array(cant)
    valor=np.where(cant==max(cant))
    valor1=np.where(valor[0]==5)
    valor2=np.where(valor[0]==3)

    valor6=np.where(valor[0]==0)
    valor7=np.where(valor[0]==1)
    valor8=np.where(valor[0]==2)
    valor9=np.where(valor[0]==4)
 
    su=len(valor2[0])
    su1=len(valor1[0])
    su2=su+su1

    su6=len(valor6[0])
    su7=len(valor7[0])
    su8=len(valor8[0])
    su9=len(valor9[0])
    # 0-angry, 1-disgust, 2-fear, 3-happy, 4-sad, 5-surprise
    if len(valor[0]) == 1:
        if valor[0] == 0:
            frame1.append(8)
        if valor[0] == 1:
            frame1.append(8)
        if valor[0] == 2:
            frame1.append(8)
        if valor[0] == 3:
            frame1.append(6)
        if valor[0] == 4:
            frame1.append(8)
        if valor[0] == 5:
            frame1.append(7)
    # 6-positive, 7-neutro, 8-negative
    if len(valor[0]) == 2:

        if su1==1:
            if su==0:
               frame1.append(8)     
        if su==1:
            if su1==1:
                frame1.append(6)
            if su1==0: 
                frame1.append(7)
        if su2==0:
            frame1.append(8)

    if len(valor[0]) == 3:
        if su2==0 or su2==1:
            frame1.append(8)
        if su2==2:
            frame1.append(7)
        
    if len(valor[0]) == 4:
        frame1.append(8)
    if len(valor[0]) == 5:
        frame1.append(8)	
    if len(valor[0]) == 6:
        frame1.append(8)
    return frame1


def emocion_escena(emo,frame1):
  c0=0
  c1=0
  c2=0
  c3=0
  c4=0
  c5=0
  c6=0
  c7=0
  c8=0
  for i in range(len(emo)):
    if emo[i] == 0:
      c0=c0+1
    if emo[i] == 1:
      c1=c1+1
    if emo[i] == 2:
      c2=c2+1
    if emo[i] == 3:
      c3=c3+1
    if emo[i] == 4:
      c4=c4+1
    if emo[i] == 5:
      c5=c5+1
    if emo[i] == 6:
      c6=c6+1
    if emo[i] == 7:
      c7=c7+1
    if emo[i] == 8:
      c8=c8+1
  cant=[c0,c1,c2,c3,c4,c5,c6,c7,c8]
  cant=np.array(cant)
  valor=np.where(cant==max(cant))
  valor1=np.where(valor[0]==5)
  valor2=np.where(valor[0]==3)
  valor3=np.where(valor[0]==6)
  valor4=np.where(valor[0]==7)
  valor5=np.where(valor[0]==8)
  valor6=np.where(valor[0]==0)
  valor7=np.where(valor[0]==1)
  valor8=np.where(valor[0]==2)
  valor9=np.where(valor[0]==4)
  
  su=len(valor2[0])
  su1=len(valor1[0])
  su2=su+su1
  su3=len(valor3[0])
  su4=len(valor4[0])
  su5=len(valor5[0])
  su6=len(valor6[0])
  su7=len(valor7[0])
  su8=len(valor8[0])
  su9=len(valor9[0])
  # 0-angry, 1-disgust, 2-fear, 3-happy, 4-sad, 5-surprise
  # 6-positive, 7-neutro, 8-negative
  if len(valor[0]) == 1:
    if valor[0] == 0:
      frame1.append(0)
    if valor[0] == 1:
      frame1.append(1)
    if valor[0] == 2:
      frame1.append(2)
    if valor[0] == 3:
      frame1.append(3)
    if valor[0] == 4:
      frame1.append(4)
    if valor[0] == 5:
      frame1.append(5)
    if valor[0] == 6:
      frame1.append(6)
    if valor[0] == 7:
      frame1.append(7)
    if valor[0] == 8:
      frame1.append(8)
    # 6-positive, 7-neutro, 8-negative
    # 0-angry, 1-disgust, 2-fear, 3-happy, 4-sad, 5-surprise
  if len(valor[0]) == 2:
    if su3+su4+su5==0:
      if su2 != 2 and su != 1:
        frame1.append(8)
      if su==1:
        frame1.append(7)
    if su4+su5 == 2:
      frame1.append(8)
    if su4+su5==1:
      if su3 != 1 and su != 1 and su1 != 1:
        frame1.append(8)
    if su5==1 and su1==1:
      frame1.append(8)
    if su3==1 and su2==2:
      frame1.append(7)
    if su==1 and su5==1:
      frame1.append(7)
    if su1==1 and su4==2:
      frame1.append(7)
    if su2==2:
      frame1.append(6)
    if su1==1 and su3==1:
      frame1.append(6)
    if su3==1 and su4==1:
      frame1.append(6)
    if su==1 and su3==1:
      frame1.append(6)
    if su==1 and su4==1:
      frame1.append(6)
    # 6-positive, 7-neutro, 8-negative
    # 0-angry, 1-disgust, 2-fear, 3-happy, 4-sad, 5-surprise
  if len(valor[0]) == 3:
    if su3+su4+su5==0 and su1==0 or su3+su4+su5==0 and su1==1 and su==0:
      frame1.append(8)
    if su5==1 and su6+su7+su8+su9 == 1:
      frame1.append(8)
    if su4==1 and su6+su7+su8+su9==2:
      frame1.append(8)
    if su4==1 and su1==1 and su6+su7+su8+su9==1:
      frame1.append(8)
    if su3+su4+su5==0 and su2==2:
      frame1.append(7)
    if su6+su7+su8+su9==1 and su==1 and su4==1:
      frame1.append(7)
    if su6+su7+su8+su9==2 and su3==1:
      frame1.append(7)
    if su3+su4==2 and su6+su7+su8+su9==1:
      frame1.append(7)
    if su6+su7+su8+su9==1 and su1+su3==2:
      frame1.append(7)
    if su2==2 and su5==1:
      frame1.append(7)
    if su1==1 and su3==1 and su5==1:
      frame1.append(7)
    if su1==1 and su4==1 and su5==1:
      frame1.append(7)
    if su==1 and su4==1 and su5==1:
      frame1.append(7)
    if su6+su7+su8+su9==1 and su==1 and su3==1:
      frame1.append(6)
    if su==1 and su3+su4==2:
      frame1.append(6)
    if su==1 and su3+su5==2:
      frame1.append(6)
    if su2==2 and su4==1:
      frame1.append(6)
    if su2==2 and su3==1:
      frame1.append(6)

  if len(valor[0]) == 4:
    if su6+su7+su8+su9+su2==4:
      frame1.append(8)
    if su6+su7+su8+su9+su==3 and su4+su5==1:
      frame1.append(8)
    if su6+su7+su8+su9+su1==2 and su5==1:
      frame1.append(8)
    if (su7!=1 and su8!=1 and su9!=1) or (su6!=1 and su7!=1 and su9!=1) or (su6!=1 and su8!=1 and su9!=1):
      if su6+su7+su8+su9+su1==2 and su5==1:
        frame1.append(8)
      if su6+su7+su8+su9==2 and su1==1 and su5==1:
        frame1.append(8)
      if su6+su7+su8+su9==2 and su1==1 and su4==1:
        frame1.append(8)
      if su3+su4+su5==3 and su6+su7+su8+su9==1:
        frame1.append(8)
      if su3==1 and su6+su7+su8+su9+su2==3:
        frame1.append(7)
      if su2==1 and su6+su7+su8+su9==1 and su3+su4==2:
        frame1.append(7)
      if su6+su7+su8+su9==1 and su2==2 and su4==1:
        frame1.append(7)
      if su4+su5==2:
        if su1+su3==2:
          frame1.append(7)
        if su6+su3==2:
          frame1.append(7)
        if su6+su7+su8+su9==1 and su==2:
          frame1.append(7)
      if su3+su5==2 and su==1:
        frame1.append(7)
      if su9+su==2 and su3+su4==2:
        frame1.append(7)
      if su4+su5==2 and su==1 and su7==1:
        frame1.append(7)
      if su6+su7==2 and su1==1 and su3==1:
        frame1.append(7)
      if su6+su7+su8+su9==1 and su2==2 and su3==1:
        frame1.append(6)
      if su2==2 and su3==1 and su5==1:
        frame1.append(6)
      if su3+su4+su5==3 and su==1:
        frame1.append(6)
      if su2==2 and su3+su4==2:
        frame1.append(6)
  
  if len(valor[0]) == 5:
    if su6+su7+su8+su9+su2==5:
      frame1.append(8)
    if su6+su7+su8+su9+su2==4 and su4==1:
      frame1.append(8)
    if su6+su7+su8+su9+su2==4 and su5==1:
      frame1.append(8)
    if su6+su7+su8+su9+su2==4 and su3==1:
      frame1.append(7)
    if su6+su7+su8+su9+su2==3:
      if su2==0 or su2==1:
        if su4+su5==2 or su3+su5==2:
          frame1.append(8)
        if su3+su4==2:
          frame1.append(7)
      if su2==2:
        frame1.append(7)
    if su6+su7+su8+su9+su2==2:
      if su1==1:
        if su==0:
          frame1.append(8)     
      if su==1:
        if su1==1:
          frame1.append(6)
        if su1==0:
          frame1.append(7)
      if su2==0:
        frame1.append(8)
  if len(valor[0]) == 6:
    if su6+su7+su8+su9+su2==6:
      frame1.append(8)
    if su6+su7+su8+su9+su2==5 and su4+su5==1:
      frame1.append(8)
    if su6+su7+su8+su9+su2==5 and su3==1:
      frame1.append(7)
    if su6+su7+su8+su9+su2==4 and su4+su5==2:
      frame1.append(8)
    if su6+su7+su8+su9+su2==4 and su3+su5==2:
      frame1.append(8)
    if su6+su7+su8+su9+su2==4 and su3+su4==2:
      frame1.append(7)
    if su6+su7+su8+su9+su2==3:
      if su2==0 or su2==1:
        frame1.append(8)
      if su2==2:
        frame1.append(7)
    else:
      frame1.append(8)    
  if len(valor[0]) == 7:
    if su3==1:
      frame1.append(7)
    if su4==1 or su5==1:
      frame1.append(8)
    if su3+su4==2:
      frame1.append(7)
    if su4+su5==2:
      frame1.append(8)
    if su3+su5==2:
      frame1.append(8)
    if su3+su4+su5==3:
      frame1.append(8)
    else:
      frame1.append(8)
  if len(valor[0]) == 8:
    if su3+su4==2:
      frame1.append(7)
    if su4+su5==2:
      frame1.append(8)
    if su3+su4+su5==3:
      frame1.append(8)
    if su3+su5==2:
      frame1.append(8)
    else:
      frame1.append(8)
  if len(valor[0]) == 9:
    frame1.append(8)
  return frame1


def emocion_final(e_escena):
    if e_escena[-1] == 0:
        n='Enojado'
        m=0
    elif e_escena[-1] == 1:
        n='Disgusto'
        m=1
    elif e_escena[-1] == 2:
        n='Miedo'
        m=2
    elif e_escena[-1] == 3:
        n='Feliz'
        m=3
    elif e_escena[-1] == 4:
        n='Triste'
        m=4
    elif e_escena[-1] == 5:
        n='Sorpresa'
        m=5
    elif e_escena[-1] == 6:
        n='E. Positiva'
        m=6
    elif e_escena[-1] == 7:
        n='E. Neutra'
        m=7
    elif e_escena[-1] == 8:
        n='E. Negativa'
        m=8
    return n


prototxtPath = r"/home/marco/pepper_sim_ws/src/Images/deploy.prototxt"
weightsPath = r"/home/marco/pepper_sim_ws/src/Images/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNet(prototxtPath, weightsPath)


nb_class = 6
hidden_dim = 512
inputs = tf.keras.Input(shape=(224, 224, 3))
custom_vgg_model=tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet',input_shape=(224, 224, 3))
last_layer = custom_vgg_model.get_layer('block5_pool').output
x = tf.keras.layers.Flatten(name='flatten')(last_layer)
x = tf.keras.layers.Dense(hidden_dim, activation=tf.nn.relu, name='fc6')(x)
x = tf.keras.layers.Dropout(.3)(x)
x = tf.keras.layers.Dense(hidden_dim, activation=tf.nn.relu, name='fc7')(x)
x = tf.keras.layers.Dropout(.3)(x)
x = tf.keras.layers.Dense(nb_class, activation=tf.nn.softmax, name='fc8')(x)
custom_vgg_model = tf.keras.Model(custom_vgg_model.input, x)

custom_vgg_model.load_weights('/home/marco/pepper_sim_ws/src/Images/model-ros.2.h5')
print("Loaded model from disk")


# Initialize the ROS Node named 'group_emotion_detection'
rospy.init_node('group_emotion_detection', anonymous=True)
# Initialize the CvBridge class
bridge = CvBridge()


CATEGORIES = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise']
count = 0
required_size=(224, 224)
fr= 20
j = 0
i1 = 0
area=[]
areas=[]
area2=[]
nframe=[]
armax=[]
lista=[]
lista1=[]
listo=[]
listo1=[]
f=1
g=1
esce=1
vere=1
emo=[]
frame1=[]
emociones=[]
nframe=[]
frame2=[]
frame3=[]
frame4=[]
frame5=[]
escenas_1=[]
emocion_f=[]



# Define a callback for the Image message
def image_callback(img_msg):
  global frame1,fr,j,i1,area,areas,area2,nframe,armax,lista,lista1,listo,listo1,f,g,esce,vere,emo,frame1,emociones,nframe,frame3,frame4

  # Try to convert the ROS Image message to a CV2 Image
  try:
    cv_image = bridge.imgmsg_to_cv2(img_msg, "bgr8")
    i1 = i1 + 1
  except :
    rospy.logerr("CvBridge Error:")
      
  frame = cv_image
  frame =  imutils.resize(frame, width=640)
  blob = cv2.dnn.blobFromImage(frame, 1, (224,224))
  net.setInput(blob)
  out = net.forward()
  k=10000
  khe2=i1
  cv2.putText(frame,str(khe2), (500, 450), cv2.FONT_ITALIC, 0.75, (255, 0, 0), 2)

  for i in range(0, out.shape[2]):
    if out[0, 0, i, 2] > 0.5:
      box = out[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
      (Xi, Yi, Xf, Yf) = box.astype("int")
      if Xi < 0: Xi = 0
      if Yi < 0: Yi = 0
      face = frame[Yi:Yf, Xi:Xf]
      
      if Yf-Yi > 0.001:
        khe=i1
        cv2.rectangle(frame, (Xi, Yi),(Xf, Yf), (0,0,255),3)
        face = cv2.resize(face, (224, 224), interpolation = cv2.INTER_AREA)
        xt = np.asarray(face)
        xt = preprocess_input(xt)
        xt = np.expand_dims(xt,axis=0)
        preds = custom_vgg_model.predict(xt)
        # print(preds)
        if np.amax(preds) > 0.9:
          cv2.putText(frame, str(CATEGORIES[np.argmax(preds)]), (Xi+5, Yi-15),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
          ar=(Yf-Yi)*(Xf-Xi)
          areas=min(k,ar)
          k=areas
          # print('Frames', khe, khe2)
          frame1=emocion_frame(khe,khe2,emo,frame1, preds)
  area2.append(areas)  

  esce,j,fr,f,g=escenas(j,fr,area2,armax,lista, lista1,listo,listo1,f,g,esce)
    
  cv2.putText(frame,str('Escena'), (30, 350), cv2.FONT_ITALIC, 0.75, (255, 0, 0), 2)
  cv2.putText(frame,str(esce), (140, 350), cv2.FONT_ITALIC, 0.75, (255, 0, 0), 2)

  if len(frame1)>0:
    frame2=frame1[-1]
    cv2.putText(frame,str('Emocion del frame:'), (30, 400), cv2.FONT_ITALIC, 0.75, (255, 0, 0), 2)
    n2=emocion_final(frame1)
    emocion_f.append(n2)
    cv2.putText(frame,str(n2), (310, 400), cv2.FONT_ITALIC, 0.75, (255, 0, 0), 2)
    escenas_1.append(esce)
    frame3.append(frame2)    
    if esce==vere:
      e_escena=emocion_escena(frame3,frame4)
      cv2.putText(frame,str('Emocion de la escena:'), (30, 450), cv2.FONT_ITALIC, 0.75, (255, 0, 0), 2)
      try:
        n1=emocion_final(e_escena)
      except:
        print("An exception occurred") # 6-positive, 7-neutro, 8-negative
      cv2.putText(frame,str(n1), (310, 450), cv2.FONT_ITALIC, 0.75, (255, 0, 0), 2)
    if esce!=vere:
      frame3=[]
    vere=esce
  emo=[]
  frame1=[]
  frame4=[]
   
  cv2.imshow("Image window",frame)
  cv2.waitKey(1)

# Initalize a subscriber to the "/camera/rgb/image_raw" topic with the function "image_callback" as a callback
sub_image = rospy.Subscriber("/pepper/camera/front/image_raw", Image, image_callback)

# Loop to keep the program from shutting down unless ROS is shut down, or CTRL+C is pressed
while not rospy.is_shutdown():
  rospy.spin()
