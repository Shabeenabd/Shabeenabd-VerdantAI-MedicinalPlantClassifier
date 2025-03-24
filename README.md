# Medicinal Plant Identification And Information Retrieval System

A comprehensive end-to-end image classification system using Vision Transformer (ViT) architecture, FastAPI backend, and Drupal frontend. This project demonstrates the implementation of modern deep learning techniques for image classification with a production-ready user-friendly web interface.

## 🌟 Table of Contents

- [Project Overview](#-project-overview)
- [Technologies Used](#-technologies-used)
- [Installation Instructions](#-installation-instructions)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [1. Model Training](#1-model-training)
  - [2. Setting up the FastAPI Backend](#2-setup-fastapi-backend)
  - [3. Analysing deployed model](#3-feedback-system-rlhf)
  - [4. Scraping image data](#4-scraper)
- [Model Performance](#-model-performance)

---



## 🔍 Project Overview 

This project consists of the following components:

1. **Building Image Classification Model :** 
   - A deep learning model based on Vision Transformer (ViT) architecture for image classification.
   - The model is trained on a large dataset of different classes of plants.

2. **Backend Server (FastAPI) :**
   - A RESTful API built with FastAPI to serve the trained Vision Transformer model.
   - The API accepts image inputs, processes them using the trained model, and returns predictions.

3. **Frontend (Drupal Application) :**
   - A web interface built with Drupal where users can upload images for classification.
   - Displays the model’s predictions and provides an interactive user experience.
   - Displays comprehensive details on various plants, including species information, characteristics, and growth habits.

4. **Android Mobile Application (PWA) :**
   - PWA Android App built from a Drupal web app using Ionic and Capacitor for cross-platform functionality.
   - Offline Support with service workers for seamless usage without an internet connection.
   - Push Notifications powered by Capacitor to keep users engaged with real-time updates.

5. **Feedback System (RLHF) :**
   - A streamlit application to monitor how well the deployed model is performing in real-world scenarios.
   - This collected data (misclassified images, model predictions, and feedback) can be used to retrain the model, ensuring that the model improves over time with new data.
   - The application provides a clear and interactive view of both successful and failed predictions, helping to identify specific areas where the model might need improvement.

6. **Scraper :**
   - This scraper is a modified version of the Bing Downloader, designed to easily scrape Images of plants.
---

## 🔧 Technologies Used

##### Deep Learning Framework : PyTorch
##### Model Architecture : Vision Transformer (ViT)
##### Backend : FastAPI
##### Frontend : Drupal 9
##### Containerization : Docker
##### CI/CD : GitLab CI/CD
##### Dashboard : Streamlit
##### Database : MongoDB
##### Experiment Tracking : Neptune AI
---

## 📋 Installation Instructions

To get started with this project, follow the instructions below to set up the environment and install the necessary dependencies.

### Prerequisites

- Python 3.7 or higher
- Node.js (for Drupal setup)
- Docker (optional, for easy deployment)
- CUDA-capable GPU (for training)

### Step 1: Clone the Repository

```bash
git clone https://gitlab.com/icfoss/Malayalam-Computing/medicinal-plant-identification-and-information-retrieval-system.git
cd medicinal-plant-identification-and-information-retrieval-system
```
### Step 2: Set Up Python Environment
1. Create a virtual environment:
```bash
python3 -m venv venv
```
2. Activate the virtual environment:
```bash
# For Windows
venv\Scripts\activate
# For macOS/Linux:
source venv/bin/activate

```
3. Install the required Python dependencies:
```bash
pip install -r requirements.txt
```

## 🎯 Project Structure

```bash
your-repository/
│
├── Drupal Application/        # Drupal frontend application
│   ├── modules/
│   ├── profiles/
│   ├── sites/
│   └── themes/   
│   
├── FastAPI Backend/
│   ├── app.py                 # FastAPI backend API implementation
│   ├── main.py                # Model loading and inference logic
│   ├── database_logging.py    # Logging User input and model prediction to database
│   └── plants_data.csv        # Mapped class id of every plants
│
├── Feedback System/
│   └── database_viewer.py     # Streamlit application to analyse model performance (RLHF)         
│  
├── Model Codes               # Model building and testing codes 
│   ├── data_augmentor.py      # Data augmentation for increasing dataset diversity
│   ├── data_split.py          # Train-test splitting
│   ├── list_classes.py        # Analysing target classes
│   ├── model_build.py         # Model training code
│   └── model_tester.ipynb     # Model testing code
│
├── Progressive Web App        # Drupal PWA Android application 
│   └── capacitor.config.json  # Configure settings for building and running a PWA app
│
├── Scraper/
│   ├── bing.py
│   ├── downloader.py
│   └── data_collector.py    # Image scraper. 
│
├── Images
│
├── requirements.txt         # Main Python dependencies for the entire project
│
└── README.md                # This README file
```

## 🚀 Getting Started
#### 1. Model Training
```bash
cd Model Codes
# create a folder for training dataset.
mkdir Dataset
# setup and load image data in subfolder structure.

Model Codes/
│
├── Dataset/
│      ├── Class 1
│      │      ├── image1.jpg
│      │      ├── image2.jpg     
│      ├── Class 2
│      .
│      .
│      .
│      └── Class n

# list all the classes and get an overview of the dataset.
python list_classes.py
#run python script for building the model.
python model_build.py
```
#### 2. Setup Fastapi backend
```bash
cd FastAPI Backend 
# edit the model path in main.py for loading the trained model in the backend server
# run Fastapi application with gunicorn server
guicorn app:app
```
#### 3. Feedback System (RLHF)
```bash
cd Feedback System
# run the streamlit application and open it in the browser.
steamlit run database_viewer.py
```
#### 4. Scraper
This is modified bing downloader for downloading plant images from bing.
```bash
cd Scraper
python data_collector.py
```
## 📊 Model Performance

- Training Accuracy: 99%
- Validation Accuracy: 93%
- Inference Time: <100ms
- Supported Image Formats: JPG, PNG, WebP
