# **uFOIL: An Unsupervised Fusion of Image and Language Understanding**

This repository contains the implementation code and pipeline for **uFOIL**, a novel ensemble-based unsupervised learning framework for automating information extraction from exam scripts. 

## **Methodology**

Our methodology combines several advanced techniques, including dynamic contrast adjustment, rotation correction, BM3D denoising, and GAN-based augmentation. We also implement a multi-step segmentation process to isolate different sections of exam scripts and use an ensemble of OCR models to extract text data.

![Methodology](/fig/methodology.png)

## **Dataset**

The dataset used in this project was sourced from 412 exam script images collected from a United International University, Bangladesh. These scripts include both handwritten and printed student details, as well as question-wise marks presented in a tabular format.


## **Preprocessing**

The detailed implementation of our preprocessing pipeline includes the following steps:
- **BM3D Denoising**: Reduces noise from exam scripts while preserving important features [💻](preproc/bm3d_filter.py).
- **Dynamic CLAHE**: Improves contrast dynamically based on local regions of the image [💻](preproc/dynamic_CLAHE.py).
- **GAN-Based Augmentation**: Generates synthetic data to expand the dataset for more robust model training [💻](preproc/gan_aug.py).
- **Rotation Correction**: Corrects for any skew in the scanned exam scripts based on detected text regions [💻](preproc/rotation_corr.py).


## **Segmentation**

Our segmentation pipeline consists of several steps:

- **Label Detection**: Detects labels like "Name", "ID", and "Course Code" using OCR [💻](segmentation/label_detection.py).
- **Section Separation**: Divides the exam script into upper and lower sections [💻](segmentation/section_separation.py).
- **Table Segmentation**: Segments the table containing question-wise marks from the lower section of the script [💻](segmentation/table_segmentation.py).



## **Text Processing and OCR**

For text extraction and validation, we use an ensemble of OCR models, including Tesseract, EasyOCR, CRAFT, and TrOCR. 

- **OCR Models**: Code to integrate various OCR models can be found [💻](text_proc/ocr_models.py).
- **Majority Voting**: After OCR, majority voting is applied to select the most accurate text output [💻](text_proc/majority_voting.py).
- **Post-Processing**: Cleans and formats the extracted text for easier validation [💻](text_proc/post_processing.py).
- **Validation**: The extracted data is validated against expected formats, such as student IDs and marks [💻](text_proc/validation.py).


## **Requirements**

To run the project, the following key Python packages are required:

- OpenCV
- PyTorch
- scikit-image
- Tesseract-OCR
- EasyOCR
- NumPy
- Matplotlib
- Pandas


## **Citation**
Will be added soon

## **Acknowledgments**

We would like to thank the Director of the Masters in Computer Science and Engineering (MSCSE) program for providing access to the exam scripts dataset.
