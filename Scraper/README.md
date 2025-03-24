# Image Downloader Web Scraper

This is a **web scraper** designed to **download images** quickly for training **image classification models**. The scraper was built by modifying the **Bing Downloader** Python module to allow fast and efficient image retrieval, making it easy to collect a large number of images for machine learning tasks.

## Features
- **Fast Image Downloading**: Downloads images from Bing Search efficiently.
- **Customizable Search Parameters**: Specify the search query, image resolution, and number of images to download.
- **Simple to Use**: Run the script and get images in a specified folder for easy integration into model training.
- **Image Classification Dataset Creation**: Ideal for generating datasets for computer vision tasks like image classification.

## Usage
1. Update the search query and other parameters in the **data_collector.py** script (name of the plants and number of images to download).
**Or** specify the parameters on the Run.

2. Run the script:
```bash
cd Scraper
python data_collector.py
```
![alt text](Artifacts/Screenshot_from_2024-12-27_12-59-25.png)

3. Images will be saved to the Dataset folder, ready to be used for training your image classification model.