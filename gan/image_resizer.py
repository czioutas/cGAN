## image_resizer.py
# Importing required libraries
import os
import sys
import numpy as np
from PIL import Image

IMAGE_SIZE = 128
IMAGE_CHANNELS = 3
IMAGE_DIR = ".\dataset\images\images\Frida_Kahlo"

# Defining image dir path.
images_path = IMAGE_DIR

training_data = []

# Iterating over the images inside the directory and resizing them using
# Pillow's resize method.
print('resizing...')

for filename in os.listdir(images_path):
    path = os.path.join(images_path, filename)
    image = Image.open(path).resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)

    training_data.append(np.asarray(image))

training_data = np.reshape(
    training_data, (-1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS))
training_data = training_data / 127.5 - 1

print('saving file...')
np.save('resized_data.npy', training_data)
