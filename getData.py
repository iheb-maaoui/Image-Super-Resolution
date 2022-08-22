import cv2
import numpy as np
import os
def read_dataset_fromKaggle(link):

  os.system('mkdir ~/.kaggle')
  os.system('cp kaggle.json ~/.kaggle/')
  os.system('chmod 600 ~/.kaggle/kaggle.json')
  os.system(f'kaggle datasets download ${link}')

  dataset_name = link.split('/')[-1]
  zip_path = os.path.join(os.getcwd(),dataset_name+'.zip')
  os.system(f'unzip {zip_path}')


def map_func(batch):
  img_lrs = []
  img_hrs = []
  for img_name in batch:
    img_name = TRAIN_DIR+img_name
    img = cv2.imread(img_name)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_lr = cv2.resize(img,(32,32),interpolation = cv2.INTER_NEAREST)
    img_hr = cv2.resize(img,(128,128),interpolation = cv2.INTER_NEAREST)
    img_lrs.append(img_lr)
    img_hrs.append(img_hr)
  img_lrs = np.array(img_lrs)
  img_hrs = np.array(img_hrs)
  return img_lrs,img_hrs


TRAIN_DIR = '/content/DIV2K_train_HR/DIV2K_train_HR/'
read_dataset_fromKaggle('sharansmenon/div2k')

images_list = os.listdir(TRAIN_DIR)
img_lr,img_hr = map_func(images_list)

path_lr = 'img_lrs'
np.save(path_lr, img_lr)

path_hr = 'img_hrs'
np.save(path_hr, img_hr)