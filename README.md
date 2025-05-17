# Digital Watermarking

Implementation of frequency domain watermarking algorithms (DCT) for copyright protection.

Reference repository https://github.com/Messi-Q/python-watermark/tree/master/image_digital_watermark/case3.

Source code was edited to be able to embed and extract the signature as an image.

### Install all dependencies:

`pip install -r requirements.txt`

## Enhanced GUI Application

A comprehensive graphical user interface has been created to make it easy to use all watermarking features:

### Running the Enhanced GUI

1. Run the application by double-clicking the `run_watermark.bat` file on Windows or use:
   ```
   python enhanced_watermark_gui.py
   ```

2. The enhanced GUI offers:
   - A modern interface with image previews
   - DCT watermarking algorithm
   - Embedding watermarks into cover images
   - Extracting watermarks from watermarked images
   - Applying various attacks to test watermark robustness
   - All results are saved in the `output` directory

## Features

### Embedding Watermarks
- Select a cover image and a signature image
- Process and save the watermarked image using DCT algorithm
- View real-time previews of both images

### Extracting Watermarks
- Select a previously watermarked image
- Extract and view the hidden watermark

### Attack Simulation
- Apply various attacks to test watermark robustness:
  - Blur, rotation, cropping, grayscale conversion
  - Noise addition, line insertion, brightness changes
  - Resizing operations

## Command Line Interface

You can also use the original command line interface:

### 1. Embedding watermark into a cover:

`python main.py --origin path_cover_image --ouput path_output_image`

Example:

> `python main.py --origin cover.jpg --ouput watermarked.jpg`
> 1. Then choice "DCT".
> 2. After that, choice "embedding".

### 2. Extracting watermark from a watermarked image:

`python main.py --origin path_watermarked_image --ouput path_extracted_signature`

Example:

> `python main.py --origin watermarked.jpg --ouput signature.jpg`
> 1. Then choice "DCT".
> 2. After that, choice "extracting".

### 3. Attacking a watermarked image:

`python main.py --origin path_watermarked_image --ouput path_attacked_image`

Example:

> `python main.py --origin watermarked.jpg --ouput watermarked.jpg`
> 1. Then choice "Attack".
> 2. After that, choice a type attack.
