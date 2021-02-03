import glob
import numpy as np 
import argparse 
import importlib
import os 
from os import path
import datetime

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import tensorflow as tf ; print("You have imported tensorflow version :", tf.__version__)
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

import keras 
print("Keras version : ",keras.__version__)
from tensorflow.keras.optimizers import Adam , Adagrad, Adadelta, RMSprop, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar100 , cifar10
from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications import EfficientNetB0

train_directory = '/home/aastha/imagenet/data/official_data/train/'
batch_size = 64

datagen = ImageDataGenerator(rescale=1./255.,  validation_split=0.05 ,   shear_range=0.2, zoom_range=0.2,horizontal_flip=True)
train_it = datagen.flow_from_directory( train_directory  , target_size=(224, 224), color_mode="rgb", batch_size=batch_size, class_mode="categorical", shuffle=True, seed=42, subset="training")
val_it = datagen.flow_from_directory(train_directory, target_size=(224, 224), color_mode="rgb", batch_size=batch_size, class_mode="categorical", shuffle=True, seed=42, subset="validation")


def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)
    
lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

class PrintLR(tf.keras.callbacks.Callback):
      def on_epoch_end(self, epoch, logs=None):
            model.save("b0_210203_model_continued1.h5") # the model is saved at the end of each epoch
            print('\nLearning rate for epoch {} is {}'.format(epoch + 1,model.optimizer.lr.numpy()))
            print("Loss :", logs['loss'])
            print("Accuracy :",logs['categorical_accuracy'])
            print()

callbacks = [ 
    PrintLR(),
    lr_callback,
    tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,save_best_only=True , verbose=1)
]



strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

with strategy.scope():
    if path.exists("b0_210203_model_continued.h5"):
        model = tf.keras.models.load_model("b0_210203_model_continued.h5")
        print("Loading old model"); print()
    else:
        model = EfficientNetB0(weights=None)    
        
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True , name='CategoricalCrossentropy'), 
        metrics=[tf.keras.metrics.CategoricalAccuracy()] )
    
    train_hist = model.fit_generator(
        train_it,  epochs=1, verbose=1, callbacks= callbacks,steps_per_epoch=train_it.samples // batch_size,
        validation_data= val_it,  validation_freq=1,validation_steps=val_it.samples // batch_size, shuffle=True)


    model.save("b0_210203_model_continued1.h5")

    print(); print(); print("Loss vs epochs data - \n", train_hist.history["loss"] )
    print()
    print("Accuracy vs epochs data - \n", train_hist.history["categorical_accuracy"] )
    print()
