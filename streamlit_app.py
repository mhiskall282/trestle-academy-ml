import streamlit as st
import numpy as np
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

# Ensure the correct libraries are installed
os.system('pip install opencv-python-headless tensorflow pillow matplotlib')

# Check for GPU availability
st.write("Checking for GPU availability...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    st.write(f"GPUs detected: {len(gpus)}")
else:
    st.write("No GPU detected. Using CPU.")

# Load your trained model
st.write("Loading the model...")
try:
    model = keras.models.load_model('model_hand.h5')
    st.write("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Define a function to preprocess the image
def preprocess_image(img):
    try:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        _, img_thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)  # Apply binary thresholding
        img_final = cv2.resize(img_thresh, (28, 28))  # Resize to 28x28
        img_final = np.reshape(img_final, (1, 28, 28, 1))  # Reshape for model input
        return img_final
    except Exception as e:
        st.error(f"Error during preprocessing: {e}")
        return None

# Define the mapping from prediction to letter
word_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K',
    11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U',
    21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
}

# Streamlit app title
st.title("Handwritten Letter Prediction")

# Upload image section
uploaded_file = st.file_uploader("Upload an image of a handwritten letter for prediction:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Open the uploaded image using PIL
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Convert the uploaded image to a NumPy array for OpenCV processing
        img_array = np.array(image)

        # Debug: Print image shape
        st.write("Image shape:", img_array.shape)

        # Preprocess the image for prediction
        processed_image = preprocess_image(img_array)
        if processed_image is None:
            st.error("Failed to preprocess the image. Please try again.")
            st.stop()

        # Debug: Visualize the preprocessed image
        st.write("Preprocessed Image:")
        fig, ax = plt.subplots()
        ax.imshow(processed_image[0, :, :, 0], cmap='gray')
        st.pyplot(fig)

        # Get the prediction from the model
        prediction = model.predict(processed_image)

        # Map the prediction to a letter
        predicted_class = np.argmax(prediction)
        predicted_letter = word_dict[predicted_class]

        # Display the final result
        st.success(f"Prediction: {predicted_letter}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
