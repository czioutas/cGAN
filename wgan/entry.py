import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE
from model import WGAN
from model_config import get_config
import numpy as np
import tensorflow_datasets as tfds
import sys
import pathlib
from PIL import Image

config = get_config()
model = WGAN(config)

# training_data = np.load('resized_data.npy')
# model.train(dataset)

IMAGE_DIR = "datasets/frida_kahlo/"
print(IMAGE_DIR)

### WEIRD THINGS TO RENAME AND RESIZE IMAGES - GOTO LINE 62
input_dir = os.listdir(IMAGE_DIR)
out_dir = IMAGE_DIR+"out/"
dirs = os.listdir(out_dir)

def rename():
    cnt=0;
    for item in input_dir:
        cnt +=1
        if os.path.isfile(IMAGE_DIR+item):
            im = Image.open(IMAGE_DIR+item)         
            im.save(IMAGE_DIR+"out/" + str(cnt) + ".jpg", 'JPEG', quality=90)    

def resize():
    for item in dirs:
        if os.path.isfile(IMAGE_DIR+item):
            im = Image.open(IMAGE_DIR+item)
            f, e = os.path.splitext(IMAGE_DIR+item)
            im.thumbnail((400, 400))
            im.save(item)
            # imResize.save(f + 'resized.jpg', 'JPEG', quality=90)

if (len(dirs) == 0):
    rename()
    resize()

ds_train = tf.data.Dataset.list_files(str(pathlib.Path(out_dir + "*.jpg")))

# for element in ds_train.enumerate().as_numpy_iterator():
#     print(element)

def process_path(file_path):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=1)
    label = tf.strings.split(file_path, "\\")
    label = tf.strings.substr(label, pos=0, len=1)[2]
    label = tf.strings.to_number(label, out_type=tf.int64)
    return image, label

## THIS IS WHERE THE THINGS HAPPENS 🤖
ds_train = ds_train.map(process_path).batch(config.bs)

dataset: tf.data.Dataset = ds_train

model.train(dataset)
