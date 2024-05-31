import pandas as pd
import numpy as np
import streamlit as st
from keras.models import load_model
from skimage.io import imread 
from skimage.transform import resize 
#from keras.models import load_model


@st.cache_data
def load_models():
    """
    Load models
    """
    mobilenet_model = load_model('models/skin_cancer_mobilenet.keras')
    
    return mobilenet_model

model = load_models()

decodePrediction = {0:"benign or non-cancerous", 1:"malignant or cancerous" }

def preprocessImage(image):
    img_array= imread(image) 
    img_resized = resize(img_array, (224, 224, 3))  # Ensure the resized image has 3 channels

    # Add a batch dimension
    img_resized = np.expand_dims(img_resized, axis=0)
    
    return img_resized


def main():
    st.title('Skin Cancer Classification')
    html_temp = """
    <div style="background:#051733 ;padding:10px">
    <h2 style="color:white;text-align:center;">Cancerous Lesion Detection</h2>
    </div>
    """

    st.markdown(html_temp, unsafe_allow_html = True)
    st.divider()

    html_temp2 = """
    <div style="background:#14302f ;padding:10px">
    <p style="color:white;text-align:left;">This project aims to detect cancerous skin lesions from images </p>
    <p style="color:white;text-align:left;"><b> </b><p>
    <p style="color:white;text-align:left;"> </p>
    </div>
    """
    st.markdown(html_temp2, unsafe_allow_html = True)
    st.divider()
    infraredImage = st.file_uploader("Upload an image of the skin lesion", type=["jpg", "jpeg", "png"], accept_multiple_files = False)
    
    if st.button('Identify Fault'):
        print('preprocessing image....')
        image = preprocessImage(infraredImage)
        print('image processed!!')
        st.image(infraredImage, use_column_width="auto")
        # Make prediction
        print('predicting.......')
        probabilities = model.predict(image)
        prediction = np.argmax(probabilities)
        print(prediction)
        # Get confidence score
        pred_probs = round(max(probabilities)[0]* 100, 2)


        

        # Show the prediction
        st.write("Prediction:", prediction)
        st.success('The skin lesion uploaded is '+ decodePrediction[prediction] + '. I have '+ str(pred_probs) + '% confidence in my prediction')
    
    # st.divider()
    # dataset = st.file_uploader("Upload a csv dataset of reddit comments. Comments must be in a 'comments' column", type=["csv"], accept_multiple_files = False)

    # if st.button('Identify Batch'):
    #     df = pd.read_csv(dataset)
    #     comments = df['comments']
    #     input_batch = preprocessBatch(comments)
    #     cnnPred12 = [np.argmax(prediction) for prediction in cnn_model12.predict(input_batch)]
    #     cnnPred02 = [np.argmax(prediction) for prediction in cnn_model02.predict(input_batch)]
    #     trfPred01 = [np.argmax(prediction) for prediction in transformer_model.predict(input_batch)]
    #     ensemblePred = [ensemblePrediction([cnnPred12[i],cnnPred02[i],trfPred01[i]]) for i in range(len(input_batch))]

    #     st.success("Prediction:" + str(ensemblePred))



if __name__=='__main__': 
    main()