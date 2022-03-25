from keras.preprocessing import image
import tensorflow as tf
import numpy as np
import sys

def img_tensor(path):
    img = image.load_img(path, target_size=(100, 100))
    tensor = np.expand_dims(img, axis=0)
    return np.mean(tensor, axis=3)

model = tf.keras.models.load_model("model1")

path = sys.argv[1]
img = image.load_img(path, target_size=(100, 100))
prediction = model.predict(img_tensor(path)/255)
category = ["dog", "cat"]
for count, val in enumerate(prediction[0]):
    if round(val): print("THIS IS A " + category[count].upper())