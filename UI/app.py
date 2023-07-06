import cv2
import streamlit as st
import pytesseract
from paddleocr import PaddleOCR
from ultralytics import YOLO
import numpy as np
from PIL import Image
import concurrent.futures
import asyncio


# Initialize Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Initialize PaddleOCR
ocr_paddle = PaddleOCR()

def perform_ocr(image, model):
    # Perform object detection
    results = model.predict(image)

    #list to store the OCR results
    ocr_tesseract_results = []
    ocr_paddle_results = []

    async def perform_ocr_tesseract(crop, name):
        # Perform OCR with Tesseract on the cropped image
        ocr_text_tesseract = pytesseract.image_to_string(crop)
        return (name, ocr_text_tesseract)

    async def perform_ocr_paddle(crop, name):
        # Perform OCR with PaddleOCR on the cropped image
        ocr_result_paddle = ocr_paddle.ocr(np.array(crop))
        ocr_text_paddle = '\n'.join([word_info[1][0] for line in ocr_result_paddle for word_info in line])
        return (name, ocr_text_paddle)

    async def process_result(result):
        boxes = result.boxes.cpu().numpy()
        crops = []
        names = []
        for i, box in enumerate(boxes):
            name = result.names[int(box.cls)]
            r = box.xyxy[0].astype(int)
            crop = image.crop((r[0], r[1], r[2], r[3]))
            crops.append(crop)
            names.append(name)

        # Perform OCR with Tesseract concurrently
        tesseract_tasks = [perform_ocr_tesseract(crop, name) for crop, name in zip(crops, names)]
        tesseract_results = await asyncio.gather(*tesseract_tasks)

        # Perform OCR with PaddleOCR concurrently
        paddle_tasks = [perform_ocr_paddle(crop, name) for crop, name in zip(crops, names)]
        paddle_results = await asyncio.gather(*paddle_tasks)

        return tesseract_results, paddle_results

    async def process_results(results):
        return await asyncio.gather(*[process_result(result) for result in results])

    # new event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        # Run the OCR processing asynchronously
        ocr_results = loop.run_until_complete(process_results(results))

        # Flatten the nested results
        for tesseract_results, paddle_results in ocr_results:
            ocr_tesseract_results.extend(tesseract_results)
            ocr_paddle_results.extend(paddle_results)

    finally:
        # Close the event loop to clean up resources
        loop.close()

    return ocr_tesseract_results, ocr_paddle_results



def display_ocr_results(results):
    for result in results:
        name, ocr_text = result
        st.markdown(f"<div class='ocr-results'><h3>{name}</h3><p>{ocr_text}</p></div>", unsafe_allow_html=True)


# Set page configuration
st.set_page_config(
    page_title="Invoice Process",
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
st.title("Invoice Processing")

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

    resized_image = image.resize((300, 300))

# Display the resized image
    st.image(resized_image, caption="Uploaded Image")

    ocr_tesseract_results, ocr_paddle_results = perform_ocr(image, model)

    st.header("Tesseract OCR Results:")
    display_ocr_results(ocr_tesseract_results)

    st.header("PaddleOCR Results:")
    display_ocr_results(ocr_paddle_results)
    st.success("OCR process completed!")

else:
    st.warning("Please upload an image.")

