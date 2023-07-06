# Intelligent-Document-Processing
This repository is a project that aims to extract data from invoices by making use of YOLOv8 object 
detection algorithm and OCR libraries that include Tesseract OCR and Paddle OCR. We take a user image as an
input and then show OCR generated text as an output.

## Step 1:
First, we extract the regions of interest using object detection model by predicting using our 
pre-trained model.

## Step 2:
Once we extract our regions of interest we perform OCR on those particular regions using Tesseract 
OCR and Paddle OCR. This makes sure we are only extracting the useful information from our invoices.

## Step 3:
Display the results

# Demo:


# Requirements
Make sure you have the following dependencies installed:

-cv2
-streamlit
-pytesseract
-paddleocr
-ultralytics
-numpy
-PIL

You can install these using the requirements.txt file:
`pip install -r requirements.txt`

# Usage:
1. Clone the repository: `git clone https://github.com/varunpusarla/invoice-processing.git`
2. Change the directory: `cd UI`
3. Create a virtual environment: `python -m venv env`
4. Activate the virtual environment: `.\env\Scripts\activate`
5. Install the required dependencies: `pip install -r requirements.txt`
6. Run the Streamlit application: `streamlit run app.py`

For Tesseract OCR you also need to install its setup which can be found in the following link:
https://github.com/UB-Mannheim/tesseract/wiki

# Acknowledgements:
1. This project utilizes the YOLO object detection model from the Ultralytics repository.
   For more information, please refer to the [Ultralytics](https://github.com/ultralytics/ultralytics) GitHub page.
2. The PaddleOCR library is used for OCR processing. For more information, please refer to the [PaddleOCR]([Ultralytics](https://github.com/ultralytics/ultralytics) GitHub page.
3. The Streamlit library is used to create the web application. For more information, please refer to the [Streamlit](https://docs.streamlit.io/) documentation.





