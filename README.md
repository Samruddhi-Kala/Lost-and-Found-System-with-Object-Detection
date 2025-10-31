# Lost and Found System with Object Detection

A Flask-based web application that helps manage lost and found items using computer vision and machine learning.

## Features

- Object detection using YOLOv8
- OCR capabilities using Tesseract
- Image similarity search using FAISS
- User-friendly web interface
- Admin dashboard for managing items

## Requirements

- Python 3.9+
- Tesseract OCR 5.5.0+
- Required Python packages listed in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Samruddhi-Kala/Lost-and-Found-System-with-Object-Detection.git
cd lostfound
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install Tesseract OCR:
- Windows: Download and install from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
- Linux: `sudo apt-get install tesseract-ocr`

4. Run the application:
```bash
python app.py
```

## Project Structure

```
lostfound/
├── app.py              # Main Flask application
├── models.py           # ML models and pipeline
├── db.py              # Database operations
├── requirements.txt    # Python dependencies
├── static/            # Static files (CSS, JS)
├── templates/         # HTML templates
└── uploads/           # Uploaded images directory
```

## Usage

1. Start the server
2. Access the web interface at `http://localhost:5000`
3. Use "Submit Lost Item" to report lost items
4. Use "Submit Found Item" to report found items
5. Admin interface available at `/admin`

