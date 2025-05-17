import os
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QComboBox, QFileDialog, QProgressBar,
                             QGroupBox, QRadioButton, QMessageBox, QTabWidget)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# Import watermarking modules
from dct_watermark import DCT_Watermark
from attack import Attack

# Create output directory if it doesn't exist
if not os.path.exists("output"):
    os.makedirs("output")


class WatermarkWorker(QThread):
    finished = pyqtSignal(np.ndarray, str)
    progress = pyqtSignal(int)
    error = pyqtSignal(str)

    def __init__(self, mode, algorithm, cover_path, signature_path, output_path):
        super().__init__()
        self.mode = mode  # embed, extract, attack
        self.algorithm = "DCT"  # Only DCT is supported
        self.cover_path = cover_path
        self.signature_path = signature_path
        self.output_path = output_path
        self.attack_type = None

    def set_attack_type(self, attack_type):
        self.attack_type = attack_type

    def run(self):
        try:
            if self.mode == "embed":
                # Load images
                self.progress.emit(10)
                img = cv2.imread(self.cover_path)
                wm = cv2.imread(self.signature_path, cv2.IMREAD_GRAYSCALE)
                
                self.progress.emit(30)
                # Select algorithm
                model = DCT_Watermark()
                
                self.progress.emit(50)
                # Embed watermark
                emb_img = model.embed(img, wm)
                
                self.progress.emit(80)
                # Save output
                cv2.imwrite(self.output_path, emb_img)
                
                self.progress.emit(100)
                self.finished.emit(emb_img, "Watermark embedded successfully!")

            elif self.mode == "extract":
                # Load watermarked image
                self.progress.emit(20)
                img = cv2.imread(self.cover_path)
                
                self.progress.emit(40)
                # Select algorithm
                model = DCT_Watermark()
                
                self.progress.emit(60)
                # Extract watermark
                signature = model.extract(img)
                
                self.progress.emit(80)
                # Save output
                cv2.imwrite(self.output_path, signature)
                
                self.progress.emit(100)
                self.finished.emit(signature, "Watermark extracted successfully!")

            elif self.mode == "attack":
                # Load watermarked image
                self.progress.emit(20)
                img = cv2.imread(self.cover_path)
                
                self.progress.emit(40)
                # Apply attack
                attack_map = {
                    "blur": Attack.blur,
                    "rotate180": Attack.rotate180,
                    "rotate90": Attack.rotate90,
                    "chop5": Attack.chop5,
                    "chop10": Attack.chop10,
                    "chop30": Attack.chop30,
                    "gray": Attack.gray,
                    "saltnoise": Attack.saltnoise,
                    "randline": Attack.randline,
                    "cover": Attack.cover,
                    "brighter10": Attack.brighter10,
                    "darker10": Attack.darker10,
                    "largersize": Attack.largersize,
                    "smallersize": Attack.smallersize,
                }
                att_img = attack_map[self.attack_type](img)
                
                self.progress.emit(80)
                # Save output
                cv2.imwrite(self.output_path, att_img)
                
                self.progress.emit(100)
                self.finished.emit(att_img, f"Attack '{self.attack_type}' applied successfully!")

        except Exception as e:
            self.error.emit(f"Error: {str(e)}")


class WatermarkingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Digital Watermarking Application")
        self.setGeometry(100, 100, 1000, 700)
        self.init_ui()

    def init_ui(self):
        # Main layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Tab widget
        tabs = QTabWidget()
        main_layout.addWidget(tabs)

        # Create tabs
        embed_tab = QWidget()
        extract_tab = QWidget()
        attack_tab = QWidget()

        tabs.addTab(embed_tab, "Embed Watermark")
        tabs.addTab(extract_tab, "Extract Watermark")
        tabs.addTab(attack_tab, "Attack Image")

        # Set up each tab
        self.setup_embed_tab(embed_tab)
        self.setup_extract_tab(extract_tab)
        self.setup_attack_tab(attack_tab)

    def setup_embed_tab(self, tab):
        layout = QVBoxLayout()
        tab.setLayout(layout)

        # Cover image selection
        cover_group = QGroupBox("Cover Image")
        cover_layout = QHBoxLayout()
        cover_group.setLayout(cover_layout)

        self.cover_path_label = QLabel("No file selected")
        cover_browse_btn = QPushButton("Browse...")
        cover_browse_btn.clicked.connect(lambda: self.browse_file("cover"))
        
        cover_layout.addWidget(self.cover_path_label)
        cover_layout.addWidget(cover_browse_btn)
        
        # Signature image selection
        signature_group = QGroupBox("Signature Image")
        signature_layout = QHBoxLayout()
        signature_group.setLayout(signature_layout)

        self.signature_path_label = QLabel("No file selected")
        signature_browse_btn = QPushButton("Browse...")
        signature_browse_btn.clicked.connect(lambda: self.browse_file("signature"))
        
        signature_layout.addWidget(self.signature_path_label)
        signature_layout.addWidget(signature_browse_btn)

        # Algorithm selection
        algorithm_group = QGroupBox("Watermarking Algorithm")
        algorithm_layout = QHBoxLayout()
        algorithm_group.setLayout(algorithm_layout)
        
        self.dct_radio = QRadioButton("DCT")
        self.dct_radio.setChecked(True)
        
        algorithm_layout.addWidget(self.dct_radio)

        # Image preview area
        preview_group = QGroupBox("Image Preview")
        preview_layout = QHBoxLayout()
        preview_group.setLayout(preview_layout)
        
        self.cover_preview = QLabel("Cover Image")
        self.cover_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cover_preview.setMinimumSize(300, 300)
        
        self.signature_preview = QLabel("Signature Image")
        self.signature_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.signature_preview.setMinimumSize(300, 300)
        
        preview_layout.addWidget(self.cover_preview)
        preview_layout.addWidget(self.signature_preview)

        # Embed button and progress bar
        action_group = QGroupBox("Embed Action")
        action_layout = QVBoxLayout()
        action_group.setLayout(action_layout)
        
        self.embed_progress = QProgressBar()
        embed_btn = QPushButton("Embed Watermark")
        embed_btn.clicked.connect(self.start_embedding)
        
        action_layout.addWidget(self.embed_progress)
        action_layout.addWidget(embed_btn)

        # Add all groups to main layout
        layout.addWidget(cover_group)
        layout.addWidget(signature_group)
        layout.addWidget(algorithm_group)
        layout.addWidget(preview_group)
        layout.addWidget(action_group)

        # Initialize variables
        self.cover_image_path = ""
        self.signature_image_path = ""
        self.watermarked_image = None

    def setup_extract_tab(self, tab):
        layout = QVBoxLayout()
        tab.setLayout(layout)

        # Watermarked image selection
        watermarked_group = QGroupBox("Watermarked Image")
        watermarked_layout = QHBoxLayout()
        watermarked_group.setLayout(watermarked_layout)

        self.watermarked_path_label = QLabel("No file selected")
        watermarked_browse_btn = QPushButton("Browse...")
        watermarked_browse_btn.clicked.connect(lambda: self.browse_file("watermarked"))
        
        watermarked_layout.addWidget(self.watermarked_path_label)
        watermarked_layout.addWidget(watermarked_browse_btn)

        # Algorithm selection
        algorithm_group = QGroupBox("Extraction Algorithm")
        algorithm_layout = QHBoxLayout()
        algorithm_group.setLayout(algorithm_layout)
        
        self.extract_dct_radio = QRadioButton("DCT")
        self.extract_dct_radio.setChecked(True)
        
        algorithm_layout.addWidget(self.extract_dct_radio)

        # Image preview area
        extract_preview_group = QGroupBox("Image Preview")
        extract_preview_layout = QHBoxLayout()
        extract_preview_group.setLayout(extract_preview_layout)
        
        self.watermarked_preview = QLabel("Watermarked Image")
        self.watermarked_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.watermarked_preview.setMinimumSize(300, 300)
        
        self.extracted_preview = QLabel("Extracted Signature")
        self.extracted_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.extracted_preview.setMinimumSize(300, 300)
        
        extract_preview_layout.addWidget(self.watermarked_preview)
        extract_preview_layout.addWidget(self.extracted_preview)

        # Extract button and progress bar
        extract_action_group = QGroupBox("Extract Action")
        extract_action_layout = QVBoxLayout()
        extract_action_group.setLayout(extract_action_layout)
        
        self.extract_progress = QProgressBar()
        extract_btn = QPushButton("Extract Watermark")
        extract_btn.clicked.connect(self.start_extracting)
        
        extract_action_layout.addWidget(self.extract_progress)
        extract_action_layout.addWidget(extract_btn)

        # Add all groups to main layout
        layout.addWidget(watermarked_group)
        layout.addWidget(algorithm_group)
        layout.addWidget(extract_preview_group)
        layout.addWidget(extract_action_group)

        # Initialize variables
        self.watermarked_image_path = ""
        self.extracted_signature = None

    def setup_attack_tab(self, tab):
        layout = QVBoxLayout()
        tab.setLayout(layout)

        # Image selection
        attack_image_group = QGroupBox("Watermarked Image to Attack")
        attack_image_layout = QHBoxLayout()
        attack_image_group.setLayout(attack_image_layout)

        self.attack_image_path_label = QLabel("No file selected")
        attack_image_browse_btn = QPushButton("Browse...")
        attack_image_browse_btn.clicked.connect(lambda: self.browse_file("attack_image"))
        
        attack_image_layout.addWidget(self.attack_image_path_label)
        attack_image_layout.addWidget(attack_image_browse_btn)

        # Attack type selection
        attack_type_group = QGroupBox("Attack Type")
        attack_type_layout = QVBoxLayout()
        attack_type_group.setLayout(attack_type_layout)
        
        self.attack_type_combo = QComboBox()
        attack_types = [
            "blur", "rotate180", "rotate90", "chop5", "chop10", "chop30",
            "gray", "saltnoise", "randline", "cover", "brighter10", 
            "darker10", "largersize", "smallersize"
        ]
        self.attack_type_combo.addItems(attack_types)
        
        attack_type_layout.addWidget(self.attack_type_combo)

        # Image preview area
        attack_preview_group = QGroupBox("Image Preview")
        attack_preview_layout = QHBoxLayout()
        attack_preview_group.setLayout(attack_preview_layout)
        
        self.attack_input_preview = QLabel("Original Watermarked Image")
        self.attack_input_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.attack_input_preview.setMinimumSize(300, 300)
        
        self.attack_output_preview = QLabel("Attacked Image")
        self.attack_output_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.attack_output_preview.setMinimumSize(300, 300)
        
        attack_preview_layout.addWidget(self.attack_input_preview)
        attack_preview_layout.addWidget(self.attack_output_preview)

        # Attack button and progress bar
        attack_action_group = QGroupBox("Attack Action")
        attack_action_layout = QVBoxLayout()
        attack_action_group.setLayout(attack_action_layout)
        
        self.attack_progress = QProgressBar()
        attack_btn = QPushButton("Apply Attack")
        attack_btn.clicked.connect(self.start_attacking)
        
        attack_action_layout.addWidget(self.attack_progress)
        attack_action_layout.addWidget(attack_btn)

        # Add all groups to main layout
        layout.addWidget(attack_image_group)
        layout.addWidget(attack_type_group)
        layout.addWidget(attack_preview_group)
        layout.addWidget(attack_action_group)

        # Initialize variables
        self.attack_image_path = ""
        self.attacked_image = None

    def browse_file(self, file_type):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image File", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if not file_path:
            return
            
        if file_type == "cover":
            self.cover_image_path = file_path
            self.cover_path_label.setText(os.path.basename(file_path))
            self.display_image(file_path, self.cover_preview)
            
        elif file_type == "signature":
            self.signature_image_path = file_path
            self.signature_path_label.setText(os.path.basename(file_path))
            self.display_image(file_path, self.signature_preview)
            
        elif file_type == "watermarked":
            self.watermarked_image_path = file_path
            self.watermarked_path_label.setText(os.path.basename(file_path))
            self.display_image(file_path, self.watermarked_preview)
            
        elif file_type == "attack_image":
            self.attack_image_path = file_path
            self.attack_image_path_label.setText(os.path.basename(file_path))
            self.display_image(file_path, self.attack_input_preview)

    def display_image(self, image_path, label):
        if isinstance(image_path, str) and os.path.exists(image_path):
            pixmap = QPixmap(image_path)
        elif isinstance(image_path, np.ndarray):
            height, width = image_path.shape[:2]
            
            if len(image_path.shape) == 2:  # Grayscale
                qimg = QImage(image_path.data, width, height, width, QImage.Format.Format_Grayscale8)
            else:  # BGR format
                rgb_image = cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB)
                qimg = QImage(rgb_image.data, width, height, width * 3, QImage.Format.Format_RGB888)
                
            pixmap = QPixmap.fromImage(qimg)
        else:
            return
            
        # Scale pixmap while preserving aspect ratio
        pixmap = pixmap.scaled(label.width(), label.height(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        label.setPixmap(pixmap)

    def start_embedding(self):
        if not self.cover_image_path or not self.signature_image_path:
            QMessageBox.warning(self, "Input Error", "Please select both cover and signature images")
            return
            
        output_path = os.path.join("output", "watermarked_" + os.path.basename(self.cover_image_path))
        
        # Initialize worker thread
        self.embed_worker = WatermarkWorker(
            mode="embed",
            algorithm="DCT",
            cover_path=self.cover_image_path,
            signature_path=self.signature_image_path,
            output_path=output_path
        )
        
        # Connect signals
        self.embed_worker.progress.connect(self.embed_progress.setValue)
        self.embed_worker.finished.connect(self.embedding_finished)
        self.embed_worker.error.connect(self.show_error)
        
        # Start worker
        self.embed_progress.setValue(0)
        self.embed_worker.start()

    def embedding_finished(self, result_img, message):
        self.watermarked_image = result_img
        self.display_image(result_img, self.cover_preview)
        QMessageBox.information(self, "Success", message)

    def start_extracting(self):
        if not self.watermarked_image_path:
            QMessageBox.warning(self, "Input Error", "Please select a watermarked image")
            return
            
        output_path = os.path.join("output", "extracted_signature.jpg")
        
        # Initialize worker thread
        self.extract_worker = WatermarkWorker(
            mode="extract",
            algorithm="DCT",
            cover_path=self.watermarked_image_path,
            signature_path=None,
            output_path=output_path
        )
        
        # Connect signals
        self.extract_worker.progress.connect(self.extract_progress.setValue)
        self.extract_worker.finished.connect(self.extracting_finished)
        self.extract_worker.error.connect(self.show_error)
        
        # Start worker
        self.extract_progress.setValue(0)
        self.extract_worker.start()

    def extracting_finished(self, result_img, message):
        self.extracted_signature = result_img
        self.display_image(result_img, self.extracted_preview)
        QMessageBox.information(self, "Success", message)

    def start_attacking(self):
        if not self.attack_image_path:
            QMessageBox.warning(self, "Input Error", "Please select an image to attack")
            return
            
        attack_type = self.attack_type_combo.currentText()
        output_path = os.path.join("output", f"attacked_{attack_type}_{os.path.basename(self.attack_image_path)}")
        
        # Initialize worker thread
        self.attack_worker = WatermarkWorker(
            mode="attack",
            algorithm="DCT",
            cover_path=self.attack_image_path,
            signature_path=None,
            output_path=output_path
        )
        self.attack_worker.set_attack_type(attack_type)
        
        # Connect signals
        self.attack_worker.progress.connect(self.attack_progress.setValue)
        self.attack_worker.finished.connect(self.attacking_finished)
        self.attack_worker.error.connect(self.show_error)
        
        # Start worker
        self.attack_progress.setValue(0)
        self.attack_worker.start()

    def attacking_finished(self, result_img, message):
        self.attacked_image = result_img
        self.display_image(result_img, self.attack_output_preview)
        QMessageBox.information(self, "Success", message)

    def show_error(self, error_message):
        QMessageBox.critical(self, "Error", error_message)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WatermarkingApp()
    window.show()
    sys.exit(app.exec()) 