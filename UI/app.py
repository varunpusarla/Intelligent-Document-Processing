import cv2
import datetime
import streamlit as st
import pytesseract
from paddleocr import PaddleOCR
from ultralytics import YOLO
import numpy as np

# Initialize Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'  # Update with the path to your Tesseract installation

# Initialize PaddleOCR
ocr_paddle = PaddleOCR()

# Function to perform OCR on the image
def perform_ocr(image):
    # Perform object detection
    results = model.predict(image)

    # Perform OCR with Tesseract on the cropped parts of the image
    ocr_tesseract_results = []
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for i, box in enumerate(boxes):
            name = result.names[int(box.cls)]
            r = box.xyxy[0].astype(int)
            crop = image[r[1]:r[3], r[0]:r[2]]
            ocr_text = pytesseract.image_to_string(crop)
            ocr_tesseract_results.append(f"Object:\n{name}\n ||| OCR Text (Tesseract):\n{ocr_text}\n")
            

    # Perform OCR with PaddleOCR on the cropped parts of the image
    ocr_paddle_results = []
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for i, box in enumerate(boxes):
            name = result.names[int(box.cls)]
            r = box.xyxy[0].astype(int)
            crop = image[r[1]:r[3], r[0]:r[2]]
            ocr_result = ocr_paddle.ocr(crop)
            ocr_text = '\n'.join([word_info[1][0] for line in ocr_result for word_info in line])
            ocr_paddle_results.append(f"Object:\n{name}\n ||| OCR Text (PaddleOCR):\n{ocr_text}\n")

    return ocr_tesseract_results, ocr_paddle_results


# Create the Streamlit app
st.title("OCR Process")

# Image upload section
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)

    # Display the uploaded image
    st.image(image, channels="BGR", caption="Uploaded Image")

    # Initialize the YOLO model
    model = YOLO('best.pt')

    # Perform OCR on the uploaded image
    ocr_tesseract_results, ocr_paddle_results = perform_ocr(image)

    st.header("Tesseract OCR Results:")
    for result in ocr_tesseract_results:
        st.write(result)
        st.write('---')

    st.header("PaddleOCR Results:")
    for result in ocr_paddle_results:
        st.write(result)
        st.write('---')

    st.write("OCR process completed!")
else:
    st.warning("Please upload an image.")
