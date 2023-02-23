import os
import shutil
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
from keras.models import load_model
from keras.preprocessing import image
from keras import layers
from keras import models
from keras import optimizers
import numpy as np

categories = ['cat', 'dog']

def ImageHandler(target_image_path):
    img = image.load_img(target_image_path, target_size = (150,150))
    img_array = np.array(img) / 255
    img_array = img_array.reshape((1,) + img_array.shape)

    return img_array

def main(target_img_path):
    target_img_path = target_img_path.split('/')[-1] # target image filename
    target_image_path = './static/images/usr/'+ target_img_path # target image path

    test = []
    test.append(ImageHandler(target_image_path))

    model = load_model('cats_and_dogs_small_2.h5')
    predict = model.predict(test)

    print(target_img_path + " : , Predict : "+ str(int(round(predict[0][0]))))
    
    return categories[int(round(predict[0][0]))]

    
if __name__ == "__main__":
	main()