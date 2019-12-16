## gets the files from the folder and predicts the result with some confidence number

import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import keras
import tensorflow as tf
import tensorflow_hub as hub
import itertools
import tensorflow.keras.backend as K
import PIL.Image as Image
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model
import h5py
from imageio import imread
from skimage.transform import resize
import cv2,scipy
from tqdm import tqdm
from keras.preprocessing import image
from skimage import transform
import pandas

data_root= "C://Users//Pawandeep//Desktop//imageset10Kd"

feature_extractor_url ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2"  #@param {type:"string"}
categories = ["Categorical_Boxplot", "Column_Charts", "Dendogram", "Heatmap", "Histogram", "Line_Chart", "Map", "Node-Link_Diagram", "Ordination_Scatterplot", "Pie_Chart", "Scatterplot", "Stacked_Area_Chart", "Timeseries","Noviz"]
test_img = []
probab=[]
classes=[]
fname=[]
true_labels = np.array(categories)
new_model =  tf.keras.models.load_model("multi_old60.h5", custom_objects={'KerasLayer': hub.KerasLayer})
def load(filename):
   np_image = Image.open(filename)
   #np_image = np.array(np_image)/255
   np_image = np.array(np_image).astype('float32') / 255
   np_image = transform.resize(np_image, (224, 224, 3))
   np_image = np.expand_dims(np_image, axis=0)
   print("single shape")#, np_image.shape)
   return np_image

for i in os.listdir(data_root):
    if i in "Thumbs.db":
        continue
    fname.append(i)
    loc= data_root+"//"+i
    test_img.append(load(loc))
#print("array", test_img.shape)

p=[]
for i in test_img:
    print("i")#, i.shape)
    pred= new_model.predict(i)
    p.append(pred)
for j in p:
    #print(j)
    classes.append(true_labels[np.argmax(j, axis=-1)])
    sess = tf.compat.v1.InteractiveSession()
    probab.append(tf.reduce_max(j, axis=1).eval())
    print("j+")
    sess.close()

df = pandas.DataFrame(data={"filename": fname, "classes": classes, "probability": probab})
print(df)
#df.to_csv("./old30+.csv", sep=';',index=False)



#print(fname,classes,probab)

#print(test_set.filenames)
#with tf.Session() as sess:
  # sess.run(print(tf.reduce_max(pred, axis=-1).eval()))


#print(pred[0[]])
#print(test_set.filenames)
#print(pred)
#print(test_set.classes)
#print(pred)  #to get the first index of the value which is the highest probability








