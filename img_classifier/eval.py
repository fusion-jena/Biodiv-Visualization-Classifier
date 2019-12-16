## This script produce the evalution report from test dataset

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_hub as hub
#from tqdm import tqdm
import itertools
import tensorflow.keras.backend as K
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model
tf.version.VERSION
import PIL.Image as Image
from sklearn.metrics import classification_report, confusion_matrix


data_root = "C://Users//Pawandeep//Desktop//Annotated_Image_Classes"   # test data from different categories in one folder
feature_extractor_url ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"  #@param {type:"string"}
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/ 255, validation_split=0.20)

IMAGE_SIZE= (224,224)
feature_extractor_layer = hub.KerasLayer(feature_extractor_url, input_shape=(224,224,3))

image_data = image_generator.flow_from_directory(str(data_root), target_size=IMAGE_SIZE, subset='training')
image_data_val = image_generator.flow_from_directory(str(data_root), target_size=IMAGE_SIZE, subset='validation', batch_size=100)
for image_batch, label_batch in image_data:
    print("Image batch shape: ", image_batch.shape)
    print("Label batch shape: ", label_batch.shape)
    break
for image_val_batch, label_val_batch in image_data_val:
    print("Image batch shape: ", image_val_batch.shape)
    print("Label batch shape: ", label_val_batch.shape)
    break
feature_batch = feature_extractor_layer(image_batch)
print(feature_batch.shape)

new_model = load_model('multi_old60.h5', custom_objects={'KerasLayer': hub.KerasLayer})
new_model.summary()

eval1=new_model.evaluate(image_val_batch, label_val_batch, batch_size=100)
print(eval1)

Y_test = np.argmax(label_val_batch, axis=1) # Convert one-hot to index
y_pred = new_model.predict_classes(image_val_batch)
print(classification_report(Y_test, y_pred))
print(confusion_matrix(Y_test, y_pred))



















