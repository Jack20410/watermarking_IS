# Digital Watermarking

A robust implementation of frequency domain watermarking algorithms using DCT (Discrete Cosine Transform) for copyright protection.

## Running the GUI Application

### On Windows:
```
Double-click the run_watermark.bat file
```

### On macOS/Linux:
```
chmod +x run_watermark.sh  
# Make the script executable (first time only)
./run_watermark.sh
```

These scripts will automatically:
- Install required dependencies
- Launch the application
- Handle any potential errors

## Features

- **DCT Watermarking Algorithm**: Industry-standard frequency domain technique
- **Image-based Signatures**: Support for embedding images as watermarks
- **Attack Simulation**: Test watermark robustness against common manipulations
- **User-friendly GUI**: Modern interface with real-time previews

## Overview

This project provides tools for embedding invisible watermarks into digital images to protect intellectual property and verify content authenticity. Based on the [python-watermark](https://github.com/Messi-Q/python-watermark/tree/master/image_digital_watermark/case3) project with enhancements to support image-based signatures.

All processing results are automatically saved in the `output` directory.
