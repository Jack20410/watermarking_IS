@echo off
echo Starting Enhanced Digital Watermarking Application...

:: Try to install required packages if they're not installed
pip install -r requirements.txt

:: Run the enhanced application
python enhanced_watermark_gui.py

:: If there's an error, pause to see it
if %ERRORLEVEL% NEQ 0 (
    echo An error occurred while running the application.
    echo Please check if all requirements are installed correctly.
    pause
)

pause 