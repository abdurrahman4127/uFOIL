# **uFOIL: An Unsupervised Fusion of Image and Language Understanding**

This repository contains the implementation code and pipeline for **uFOIL**, a novel ensemble-based unsupervised learning framework for automating information extraction from exam scripts. 

## **Methodology**

Our methodology combines several advanced techniques, including dynamic contrast adjustment, rotation correction, BM3D denoising, and GAN-based augmentation. We also implement a multi-step segmentation process to isolate different sections of exam scripts and use an ensemble of OCR models to extract text data.

![Methodology](/fig/methodology.png)

## **Dataset**

The dataset used in this project was sourced from 412 exam script images collected from the United International University, Bangladesh. These scripts include both handwritten and printed student details, as well as question-wise marks presented in a tabular format.


## **Preprocessing**

The detailed implementation of our preprocessing pipeline includes the following steps:
- **BM3D Denoising**: Reduces noise from exam scripts while preserving important features
- **Dynamic CLAHE**: Improves contrast dynamically based on local regions of the image
- **GAN-Based Augmentation**: Generates synthetic data to expand the dataset for more robust model training  
- **Rotation Correction**: Corrects for any skew in the scanned exam scripts based on detected text regions  


## **Segmentation**

Our segmentation pipeline consists of several steps:

- **Label Detection**: Detects labels like "Name", "ID", and "Course Code" using OCR 
- **Section Separation**: Divides the exam script into upper and lower sections  
- **Table Segmentation**: Segments the table containing question-wise marks from the lower section of the script 



## **Text Processing and OCR**

For text extraction and validation, we used an ensemble of OCR models, including Tesseract, EasyOCR, CRAFT, and TrOCR. 

- **OCR Models**
- **Majority Voting**
- **Post-Processing**
- **Validation**


## **Acknowledgments**
We would like to thank the Director of the Master's in Computer Science and Engineering (MSCSE) program at United International University for providing access to the exam scripts dataset.

## **Citation**
```bibtex
@article{rahman2025ufoil,
  title={uFOIL: An Unsupervised Fusion of Image Processing and Language Understanding},
  author={Rahman, Md Abdur and Hasan, Md Tanzimul and Howlader, Umar Farooq and Kader, Md Abdul and Islam, Md Motaharul and Pham, Phuoc Hung and Hassan, Mohammad Mehedi},
  journal={IEEE Access},
  year={2025},
  publisher={IEEE}
}
```
