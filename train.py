import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
from SRGAN import SuperResolution

img_lr = np.load(r'C:\Users\IHEB\Documents\ImageSuperResolution\img_lrs.npy')
img_hr = np.load(r'C:\Users\IHEB\Documents\ImageSuperResolution\img_hrs.npy')

img_hr = img_hr.astype(np.float32)
img_lr = img_lr.astype(np.float32)

BATCH_SIZE = 1
BUFFER_SIZE = len(img_lr)
dataset = tf.data.Dataset.from_tensor_slices(((img_lr,img_hr),np.ones((len(img_lr),1))))
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

low_resolution_shape = (32,32,3)
high_resolution_shape = (128,128,3)
super_res = SuperResolution(low_resolution_shape,high_resolution_shape,16)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'super_res.h5',
    monitor="g_loss",
    save_best_only=True,
)

callbacks = [checkpoint]

super_res.compile()

history = super_res.fit(dataset,epochs=20,batch_size = 1,shuffle=True,callbacks=callbacks)