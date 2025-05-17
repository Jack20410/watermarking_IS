#!/bin/bash
echo "Starting Enhanced Digital Watermarking Application..."

# Try to install required packages if they're not installed
pip install -r requirements.txt

# Run the enhanced application
python enhanced_watermark_gui.py

# If there's an error, display it
if [ $? -ne 0 ]; then
    echo "An error occurred while running the application."
    echo "Please check if all requirements are installed correctly."
    read -p "Press any key to continue..."
fi

read -p "Press any key to continue..." 