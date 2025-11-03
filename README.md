# Lost and Found System with Object Detection

A web application built with Flask that leverages computer vision and machine learning to streamline the management of lost and found items.

## Overview

This system enables users to report and search for lost or found items using advanced object detection (YOLOv8), OCR (Tesseract), and image similarity search (FAISS). An administrative dashboard is provided for efficient item management.

## Features

- **Object Detection:** Identify items in images using YOLOv8.
- **OCR Integration:** Extract text from images with Tesseract OCR.
- **Image Similarity Search:** Find visually similar items using FAISS.
- **User Interface:** Simple web forms for submitting and searching items.
- **Admin Dashboard:** Manage and review reported items.

## Requirements

- Python 3.9 or newer
- Tesseract OCR 5.5.0 or newer
- All Python dependencies listed in `requirements.txt`

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Samruddhi-Kala/Lost-and-Found-System-with-Object-Detection.git
   cd Lost-and-Found-System-with-Object-Detection
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Tesseract OCR**
   - **Windows:** Download and install from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
   - **Linux:** 
     ```bash
     sudo apt-get install tesseract-ocr
     ```
   - Ensure the Tesseract executable is added to your system PATH.

4. **Run the application**
   ```bash
   python app.py
   ```

## Project Structure

```
Lost-and-Found-System-with-Object-Detection/
├── app.py              # Main Flask application
├── models.py           # ML models and processing pipeline
├── db.py               # Database operations
├── requirements.txt    # Python dependencies
├── static/             # Static files (CSS, JS)
├── templates/          # HTML templates
└── uploads/            # Uploaded images
```

## Usage

1. Start the Flask server.
2. Open your browser and navigate to `http://localhost:5000`.
3. Use the "Submit Lost Item" form to report a lost item.
4. Use the "Submit Found Item" form to report a found item.
5. Access the admin dashboard at `/admin` for item management.

## Additional Notes

- Uploaded images are stored in the `uploads/` directory.
- For production deployment, configure environment variables and security settings as needed.
- Ensure all dependencies are installed and up to date for optimal performance.

## License

This project is for educational purposes. Refer to the repository for licensing details.

