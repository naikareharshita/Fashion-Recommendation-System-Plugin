import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle
from PIL import Image

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3)) #using inbuilt resnet50 model
model.trainable = False     #used already trained model no more training

#updating the model to add top layer
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# print(model.summary())

def extract_features(img_path,model):
    img = image.load_img(img_path,target_size=(224,224))    #load img from path
    img_array = image.img_to_array(img)                     #convert img to array
    expanded_img_array = np.expand_dims(img_array, axis=0)  #expand img- create batch
    preprocessed_img = preprocess_input(expanded_img_array) #perprocessing
    result = model.predict(preprocessed_img).flatten()      #predict based on model
    normalized_result = result / norm(result)               #normalization

    return normalized_result

filenames = []

for file in os.listdir('images'):
    filenames.append(os.path.join('images',file))

feature_list = []

for file in tqdm(filenames):
    feature_list.append(extract_features(file,model))

pickle.dump(feature_list,open('embeddings.pkl','wb'))
pickle.dump(filenames,open('filenames.pkl','wb'))
