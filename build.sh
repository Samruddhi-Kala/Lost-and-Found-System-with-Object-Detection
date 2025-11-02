#!/bin/bash

# Update package list
apt-get update

# Install system dependencies
apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    tesseract-ocr-eng \
    ffmpeg \
    libsm6 \
    libxext6

# Install Python packages
pip install -r requirements.txt

# Create required directories
mkdir -p uploads

# Print versions for debugging
echo "Tesseract version:"
tesseract --version

echo "Python version:"
python --version

echo "Pip packages:"
pip list