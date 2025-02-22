import streamlit as st
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model
model = load_model('model.h5')

# Define the target size for resizing images
target_size = (400, 400)

# Function to load and preprocess the image for prediction
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values
    return img_array

# Function to make predictions
def predict_image(image_path):
    # Preprocess the image
    img_array = preprocess_image(image_path)
    # Make prediction
    prediction = model.predict(img_array)
    # Get the predicted class label
    predicted_class_index = np.argmax(prediction)
    # Map the class index to the class name
    class_labels = ['EOSINOPHIL','LYMPHOCYTE','Malaria Parasitized','Malaria Uninfected','MONOCYTE','NEUTROPHIL' ]
    predicted_class_label = class_labels[predicted_class_index]
    # Get the confidence score
    confidence_score = prediction[0][predicted_class_index]
    return predicted_class_label, confidence_score

# Load a pre-trained CNN model for blood smear image classification
blood_smear_model = load_model('model.h5')

# Function to check if the uploaded image is a blood smear image
def is_blood_smear_image(image_path):
    img_array = preprocess_image(image_path)
    prediction = blood_smear_model.predict(img_array)
    if prediction[0][0] < 0.5:  # Threshold for blood smear image classification
        return True
    else:
        return False

# Streamlit UI

st.title('Malaria Detection and WBC Classification')
st.write('Upload an image')

# File uploader allows user to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Load and display the uploaded image
    img = image.load_img(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write('Click Predict Button to predict the image')

    # Check if the uploaded image is a blood smear image
    if not is_blood_smear_image(uploaded_file):
        st.error("Please upload a correct blood smear image")
    else:
        # Predict the class of the uploaded image
        if st.button('Predict'):
            # Make prediction
            predicted_class, confidence = predict_image(uploaded_file)
        
            # Display the predicted class and confidence score
            st.write(f'Predicted Class: {predicted_class}')
            st.write(f'Confidence: {confidence:.2f}')
            
            # Display the image with prediction details
            fig, ax = plt.subplots()
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f'Predicted Class: {predicted_class}\nConfidence: {confidence:.2f}')
            st.pyplot(fig)