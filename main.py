import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import base64

import pixellib
from pixellib.tune_bg import alter_bg

#basic lists of files and features
feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# st.title('Fashion Recommender System') #title
st.markdown("<h1 style='text-align: right; color: black;'>Fashion Recommender System</h1>", unsafe_allow_html=True)


def add_bg_from_local(image_file):
   with open(image_file, "rb") as image_file:
       encoded_string = base64.b64encode(image_file.read())
   st.markdown(
   f"""
   <style>
   .stApp {{
       background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
       background-size:cover
   }}
   </style>
   """,
   unsafe_allow_html=True
   )
add_bg_from_local('bg_image.avif')



#save uploaded file
def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

#feature extraction
def feature_extraction(img_path,model):
    img = image.load_img(img_path, target_size=(224, 224))  #load img from path
    img_array = image.img_to_array(img)                     #convert img to array
    expanded_img_array = np.expand_dims(img_array, axis=0)  #expand img- create batch
    preprocessed_img = preprocess_input(expanded_img_array) #perprocessing
    result = model.predict(preprocessed_img).flatten()      #predict based on model
    normalized_result = result / norm(result)               #normalization

    return normalized_result

#generate recommendations
def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean') #alternative and parallel processing can be used
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices

# steps
# 1. file upload -> save
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # if file has been uploaded
        # 2. display the file
        display_image = Image.open(uploaded_file)
        new_image = display_image.resize((300, 300))
        left_co, cent_co, last_co = st.columns(3)
        with cent_co:
            st.image(new_image, caption='Your Uploaded Image')
            st.markdown(
                """
                <style>
                img {
                    cursor: pointer;
                    transition: all .2s ease-in-out;
                }
                img:hover {
                    transform: scale(1.1);
                }
                </style>
                """,
                unsafe_allow_html=True,
            )

        # 3. Remove the background as it creates noise
        change_bg = alter_bg()
        change_bg.load_pascalvoc_model("deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")
        change_bg.color_bg(os.path.join("uploads",uploaded_file.name), colors=(255, 255, 255), output_image_name=os.path.join("uploads",uploaded_file.name))

        #  4. Feature extraction
        features = feature_extraction(os.path.join("uploads",uploaded_file.name),model)

        # 5. Recommendention
        indices = recommend(features,feature_list)

        # 6. Display recommendations
        col1,col2,col3,col4,col5 = st.columns(5)

        with col1:
            st.image(filenames[indices[0][0]])
        with col2:
            st.image(filenames[indices[0][1]])
        with col3:
            st.image(filenames[indices[0][2]])
        with col4:
            st.image(filenames[indices[0][3]])
        with col5:
            st.image(filenames[indices[0][4]])
    else:
        st.header("Some error occured in file upload")
st.markdown("<h6 style='color:black;'>Instructions</h6>", unsafe_allow_html=True)
st.markdown("<p><font color='black'>*Click on the Browse files,select the image and it will generate recommendation </p>", unsafe_allow_html=True)