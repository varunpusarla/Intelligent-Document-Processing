import cv2
import streamlit as st
import pytesseract
from paddleocr import PaddleOCR
from ultralytics import YOLO
import numpy as np
from PIL import Image

# Initialize Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Initialize PaddleOCR
ocr_paddle = PaddleOCR()

# Function to perform OCR on the image
def perform_ocr(image, model):
    # Perform object detection
    results = model.predict(image)

    # Perform OCR with Tesseract on the cropped parts of the image
    ocr_tesseract_results = []
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for i, box in enumerate(boxes):
            name = result.names[int(box.cls)]
            r = box.xyxy[0].astype(int)
            crop = image.crop((r[0], r[1], r[2], r[3]))
            ocr_text = pytesseract.image_to_string(crop)
            ocr_tesseract_results.append((name, ocr_text))

    # Perform OCR with PaddleOCR on the cropped parts of the image
    ocr_paddle_results = []
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for i, box in enumerate(boxes):
            name = result.names[int(box.cls)]
            r = box.xyxy[0].astype(int)
            crop = image.crop((r[0], r[1], r[2], r[3]))
            ocr_result = ocr_paddle.ocr(np.array(crop))
            ocr_text = '\n'.join([word_info[1][0] for line in ocr_result for word_info in line])
            ocr_paddle_results.append((name, ocr_text))

    return ocr_tesseract_results, ocr_paddle_results

def display_ocr_results(results):
    for result in results:
        name, ocr_text = result
        st.markdown(f"<div class='ocr-results'><h3>{name}</h3><p>{ocr_text}</p></div>", unsafe_allow_html=True)


# Set page configuration
st.set_page_config(
    page_title="OCR Process",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set CSS styles
st.markdown(
    """
    <style>
    .ocr-results {
        background-color: rgba(0, 0, 0, 0.6);
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 20px;
    }

    .ocr-results h3 {
        color: #ffffff;
        margin-top: 0;
        margin-bottom: 10px;
    }

    .ocr-results p {
        color: #ffffff;
        margin-top: 0;
        margin-bottom: 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Create the Streamlit app
st.title("OCR Process")

# Load the YOLO model
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

# Image upload section
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    image = Image.open(uploaded_file)
    image.thumbnail((600, 600))

    image = Image.open(uploaded_file)

# Resize the image
    resized_image = image.resize((300, 300))

# Display the resized image
    st.image(resized_image, caption="Uploaded Image")

    # Display the uploaded image
    # st.image(image, caption="Uploaded Image", use_column_width=True, width=200)

    # Perform OCR on the uploaded image
    ocr_tesseract_results, ocr_paddle_results = perform_ocr(image, model)

    # st.header("Tesseract OCR Results:")
    # for result in ocr_tesseract_results:
    #     name, ocr_text = result
    #     st.write("**Object:**", name)
    #     st.write("**OCR Text (Tesseract):**", ocr_text)
    #     st.write("---")

    # st.header("PaddleOCR Results:")
    # for result in ocr_paddle_results:
    #     name, ocr_text = result
    #     st.write("**Object:**", name)
    #     st.write("**OCR Text (PaddleOCR):**", ocr_text)
    #     st.write("---")

    st.header("Tesseract OCR Results:")
    display_ocr_results(ocr_tesseract_results)

    st.header("PaddleOCR Results:")
    display_ocr_results(ocr_paddle_results)
    st.success("OCR process completed!")

else:
    st.warning("Please upload an image.")

