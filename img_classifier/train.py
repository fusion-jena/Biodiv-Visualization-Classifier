###script to train a new model. For training each folder name is a class. In each folder are the respective training sample. Due to copyright issue, we upload our images online. Please use your own images. 
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

categories = ["Categorical_Boxplot", "Column_Charts", "Dendogram", "Heatmap", "Line_Chart", "Map", "Node-Link_Diagram", "Ordination_Scatterplot", "Pie_Chart", "Scatterplot", "Stacked_Area_Chart"]

data_root = "C://Users//Pawandeep//Desktop//Annotated_Image_new_Classes//Classes"
feature_extractor_url ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"  #@param {type:"string"}
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/ 255)
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/ 255, validation_split=0.20)
#function to load the model
def feature_extractor(x):
  feature_extractor_module = hub.Module(feature_extractor_url)
   return feature_extractor_module(x)

IMAGE_SIZE= (224,224)
#IMAGE_SIZE = hub.get_expected_image_size(hub.Module(feature_extractor_url))
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
feature_extractor_layer.trainable = True

model = tf.keras.Sequential([
     feature_extractor_layer,
     layers.Dense(image_data.num_classes, activation='softmax') ])
print(model.summary())

result = model(image_batch)
print(result.shape)

model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

class CollectBatchStats(tf.keras.callbacks.Callback):
    def __init__(self):
       self.batch_losses = []
	   self.batch_acc = []
     def on_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs['loss'])
		self.batch_acc.append(logs['acc'])


steps_per_epoch = image_data.samples // image_data.batch_size
batch_stats = CollectBatchStats()
validation_steps = image_data_val.samples / image_data_val.batch_size
# #with tf.compat.v1.Session() as sess:

model.fit((item for item in image_data), epochs=60,
              steps_per_epoch=steps_per_epoch, callbacks=[batch_stats],
              validation_data=(item for item in image_data_val), validation_steps = validation_steps, verbose=2)
eval=model.evaluate(image_val_batch, label_val_batch)
print(eval)
eval1=model.evaluate(image_batch, label_batch)
print(eval1)
model.save('model_name.h5')

new_model = load_model('multi_old60.h5', custom_objects={'KerasLayer': hub.KerasLayer})
new_model.summary()

eval1=new_model.evaluate(image_val_batch, label_val_batch, batch_size=100)
print(eval1)

Y_test = np.argmax(label_val_batch, axis=1) # Convert one-hot to index
y_pred = new_model.predict_classes(image_val_batch)
print(classification_report(Y_test, y_pred))
print(confusion_matrix(Y_test, y_pred))


#pred = new_model.predict(image_val_batch)
#print(pred)



















