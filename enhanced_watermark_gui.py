import os
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QComboBox, QFileDialog, QProgressBar,
                             QGroupBox, QRadioButton, QMessageBox, QTabWidget, QSplitter,
                             QSlider, QScrollArea)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# Import watermarking modules
from dct_watermark import DCT_Watermark
from attack import Attack
from evaluation import WatermarkEvaluator  # Import our new evaluation module

# Create output directory if it doesn't exist
if not os.path.exists("output"):
    os.makedirs("output")


class WatermarkWorker(QThread):
    finished = pyqtSignal(np.ndarray, str, np.ndarray)
    progress = pyqtSignal(int)
    error = pyqtSignal(str)

    def __init__(self, mode, algorithm, cover_path, signature_path, output_path):
        super().__init__()
        self.mode = mode  # embed, extract, attack, detect
        self.algorithm = "DCT"  # Only DCT is supported
        self.cover_path = cover_path
        self.signature_path = signature_path
        self.output_path = output_path
        self.attack_type = None
        self.sensitivity = 0.5  # Default sensitivity for detection
        self.comparison_mode = False  # Whether to compare with a sample signature

    def set_attack_type(self, attack_type):
        self.attack_type = attack_type
        
    def set_sensitivity(self, sensitivity):
        self.sensitivity = sensitivity
        
    def set_comparison_mode(self, enabled, sample_signature_path=None):
        self.comparison_mode = enabled
        self.sample_signature_path = sample_signature_path

    def run(self):
        try:
            if self.mode == "embed":
                # Load images
                self.progress.emit(10)
                img = cv2.imread(self.cover_path)
                original_img = img.copy()
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
                self.finished.emit(emb_img, f"Watermark embedded successfully using DCT!", original_img)

            elif self.mode == "extract":
                # Load watermarked image
                self.progress.emit(20)
                img = cv2.imread(self.cover_path)
                original_img = img.copy()  # Store original image for consistency with signal
                
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
                self.finished.emit(signature, f"Watermark extracted successfully using DCT!", original_img)

            elif self.mode == "detect":
                # Load image to check
                self.progress.emit(20)
                img = cv2.imread(self.cover_path)
                original_img = img.copy()
                
                self.progress.emit(40)
                # Select algorithm
                model = DCT_Watermark()
                
                self.progress.emit(60)
                # Extract potential watermark
                potential_watermark = model.extract(img)
                
                self.progress.emit(70)
                
                # If comparison mode is enabled and we have a sample signature
                if self.comparison_mode and hasattr(self, 'sample_signature_path') and self.sample_signature_path:
                    # Load the sample signature
                    sample_signature = cv2.imread(self.sample_signature_path, cv2.IMREAD_GRAYSCALE)
                    
                    # Calculate similarity between extracted watermark and sample signature
                    similarity_score = self.calculate_similarity(potential_watermark, sample_signature)
                    
                    # Adjust the score based on sensitivity
                    threshold = 1.0 - self.sensitivity  # Lower sensitivity means higher threshold
                    
                    # Create a visualization showing the comparison
                    detection_vis = self.create_comparison_visualization(potential_watermark, sample_signature, similarity_score)
                    cv2.imwrite(self.output_path, detection_vis)
                    
                    # Determine if the specific watermark is present
                    if similarity_score > threshold:
                        message = f"MATCH FOUND! Similarity: {int(similarity_score * 100)}%"
                        confidence_color = (0, 255, 0)  # Green for match
                    else:
                        message = f"NO MATCH. Similarity: {int(similarity_score * 100)}%"
                        confidence_color = (0, 0, 255)  # Red for no match
                else:
                    # Standard detection without comparison (existing code)
                    # Calculate a detection score based on the pattern of the extracted signature
                    detection_score = self.calculate_detection_score(potential_watermark)
                    
                    # Adjust the score based on sensitivity
                    threshold = 1.0 - self.sensitivity  # Lower sensitivity means higher threshold
                    
                    # Create a visualization of the detection
                    detection_vis = self.create_detection_visualization(potential_watermark, detection_score)
                    cv2.imwrite(self.output_path, detection_vis)
                    
                    # Determine if a watermark is present based on the detection score and threshold
                    if detection_score > threshold:
                        message = f"Watermark DETECTED with {int(detection_score * 100)}% confidence"
                        confidence_color = (0, 255, 0)  # Green for detected
                    else:
                        message = f"No watermark detected (confidence: {int(detection_score * 100)}%)"
                        confidence_color = (0, 0, 255)  # Red for not detected
                
                # Add confidence indicator to the visualization
                detection_vis = cv2.putText(detection_vis, message, (20, 30), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, confidence_color, 2)
                
                self.progress.emit(100)
                self.finished.emit(detection_vis, message, original_img)

            elif self.mode == "attack":
                # Load watermarked image
                self.progress.emit(20)
                img = cv2.imread(self.cover_path)
                original_img = img.copy()  # Store original image for consistency with signal
                
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
                
                self.progress.emit(60)
                # Save output
                cv2.imwrite(self.output_path, att_img)
                
                # Extract watermark from attacked image
                self.progress.emit(80)
                model = DCT_Watermark()
                
                # We don't need to save this, just for demonstration
                _ = model.extract(att_img)
                
                self.progress.emit(100)
                self.finished.emit(att_img, f"Attack '{self.attack_type}' applied successfully!", original_img)

        except Exception as e:
            self.error.emit(f"Error: {str(e)}")
            
    def calculate_similarity(self, extracted_watermark, sample_signature):
        """Calculate similarity between extracted watermark and sample signature"""
        if isinstance(extracted_watermark, list):
            # Handle the case where extract returns a list
            extracted_watermark = extracted_watermark[0]
        
        # Reshape if needed
        if len(extracted_watermark.shape) == 1:
            # Try to reshape to a square for comparison
            size = int(np.sqrt(extracted_watermark.size))
            extracted_watermark = extracted_watermark[:size*size].reshape(size, size)
                
        # Resize sample to match
        sample_signature = cv2.resize(sample_signature, 
                                     (extracted_watermark.shape[1], extracted_watermark.shape[0]), 
                                     interpolation=cv2.INTER_AREA)
        
        # Normalize both to 0-255 range if not already
        if extracted_watermark.dtype != np.uint8:
            extracted_watermark = cv2.normalize(extracted_watermark, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if sample_signature.dtype != np.uint8:
            sample_signature = cv2.normalize(sample_signature, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Apply threshold to make binary images for better comparison
        _, extracted_binary = cv2.threshold(extracted_watermark, 127, 255, cv2.THRESH_BINARY)
        _, sample_binary = cv2.threshold(sample_signature, 127, 255, cv2.THRESH_BINARY)
        
        # Calculate similarity metrics
        
        # 1. Structural similarity index (higher is better)
        try:
            from skimage.metrics import structural_similarity as ssim
            ssim_score = ssim(extracted_binary, sample_binary)
        except ImportError:
            # Fallback if scikit-image is not available
            ssim_score = 0.5  # Default value
        
        # 2. Template matching (higher is better)
        # Use multiple template matching methods and take the average
        methods = [cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED]
        template_scores = []
        for method in methods:
            result = cv2.matchTemplate(extracted_binary, sample_binary, method)
            template_scores.append(np.max(result))
        template_score = np.mean(template_scores)
        
        # 3. Feature matching using ORB
        # This helps detect if the same features are present in both images
        try:
            orb = cv2.ORB_create()
            kp1, des1 = orb.detectAndCompute(extracted_binary, None)
            kp2, des2 = orb.detectAndCompute(sample_binary, None)
            
            # If features were detected in both images
            if des1 is not None and des2 is not None and len(des1) > 0 and len(des2) > 0:
                # Create BFMatcher object
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                # Match descriptors
                matches = bf.match(des1, des2)
                
                # Calculate feature match score
                if len(matches) > 0:
                    # Sort matches by distance
                    matches = sorted(matches, key=lambda x: x.distance)
                    # Calculate average distance of top matches (lower is better)
                    top_matches = matches[:min(len(matches), 10)]
                    avg_distance = sum(m.distance for m in top_matches) / len(top_matches)
                    # Convert to a score (higher is better)
                    feature_score = 1.0 - (avg_distance / 100.0)
                    feature_score = max(0.0, min(1.0, feature_score))
                else:
                    feature_score = 0.0
            else:
                feature_score = 0.0
        except Exception:
            feature_score = 0.0
        
        # 4. Histogram comparison (higher is better)
        hist1 = cv2.calcHist([extracted_binary], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([sample_binary], [0], None, [256], [0, 256])
        hist_score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        # 5. Pixel-wise comparison (percentage of matching pixels)
        match_count = np.sum(extracted_binary == sample_binary)
        pixel_match_score = match_count / extracted_binary.size
        
        # 6. Edge comparison
        edges1 = cv2.Canny(extracted_binary, 100, 200)
        edges2 = cv2.Canny(sample_binary, 100, 200)
        edge_match_count = np.sum((edges1 > 0) & (edges2 > 0))
        edge_total = np.sum((edges1 > 0) | (edges2 > 0))
        edge_score = edge_match_count / edge_total if edge_total > 0 else 0.0
        
        # Combine scores (weighted average)
        # Give higher weight to feature matching and edge comparison as they're better at distinguishing different logos
        similarity = (
            ssim_score * 0.15 +
            template_score * 0.15 +
            feature_score * 0.3 +
            hist_score * 0.1 +
            pixel_match_score * 0.1 +
            edge_score * 0.2
        )
        
        # Apply a more aggressive sigmoid function to push scores away from the middle
        # This will make different signatures have lower scores and matching signatures have higher scores
        similarity = 1.0 / (1.0 + np.exp(-12 * (similarity - 0.6)))
        
        # Normalize to 0-1 range
        return min(max(similarity, 0.0), 1.0)
        
    def create_comparison_visualization(self, extracted_watermark, sample_signature, similarity_score):
        """Create a visualization comparing extracted watermark with sample signature"""
        if isinstance(extracted_watermark, list):
            # Handle the case where extract returns a list
            extracted_watermark = extracted_watermark[0]
        
        # Reshape if needed
        if len(extracted_watermark.shape) == 1:
            # Try to reshape to a square
            size = int(np.sqrt(extracted_watermark.size))
            extracted_watermark = extracted_watermark[:size*size].reshape(size, size)
        
        # Convert to 8-bit image if not already
        if extracted_watermark.dtype != np.uint8:
            extracted_watermark = cv2.normalize(extracted_watermark, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Resize for better visibility
        vis_size = 300
        extracted_vis = cv2.resize(extracted_watermark, (vis_size, vis_size), interpolation=cv2.INTER_NEAREST)
        
        # Resize sample signature to match
        sample_vis = cv2.resize(sample_signature, (vis_size, vis_size), interpolation=cv2.INTER_NEAREST)
        
        # Apply threshold to make binary images for better visualization
        _, extracted_binary = cv2.threshold(extracted_vis, 127, 255, cv2.THRESH_BINARY)
        _, sample_binary = cv2.threshold(sample_vis, 127, 255, cv2.THRESH_BINARY)
        
        # Apply color maps for better visualization
        extracted_color = cv2.applyColorMap(extracted_binary, cv2.COLORMAP_JET)
        sample_color = cv2.applyColorMap(sample_binary, cv2.COLORMAP_JET)
        
        # Create a difference visualization
        diff_img = cv2.absdiff(extracted_binary, sample_binary)
        diff_color = cv2.applyColorMap(diff_img, cv2.COLORMAP_HOT)
        
        # Add labels to the images
        extracted_color = cv2.putText(extracted_color, "Extracted", (10, 30), 
                                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        sample_color = cv2.putText(sample_color, "Sample", (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        diff_color = cv2.putText(diff_color, "Difference", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Create a combined visualization with all three images side by side
        # Create a horizontal stacked image
        comparison_vis = np.hstack((extracted_color, sample_color, diff_color))
        
        # Add a title bar with similarity information
        if similarity_score > 0.7:  # High similarity
            title_color = (0, 255, 0)  # Green
            border_color = (0, 255, 0)
        elif similarity_score > 0.4:  # Medium similarity
            title_color = (0, 255, 255)  # Yellow
            border_color = (0, 255, 255)
        else:  # Low similarity
            title_color = (0, 0, 255)  # Red
            border_color = (0, 0, 255)
        
        # Add a border around the visualization
        border_size = 5
        comparison_vis = cv2.copyMakeBorder(comparison_vis, border_size, border_size, border_size, border_size,
                                          cv2.BORDER_CONSTANT, value=border_color)
        
        return comparison_vis

    def calculate_detection_score(self, potential_watermark):
        """Calculate a detection score based on the pattern in the extracted potential watermark"""
        if isinstance(potential_watermark, list):
            # Handle the case where extract returns a list
            potential_watermark = potential_watermark[0]
        
        # Reshape if needed
        if len(potential_watermark.shape) == 1:
            # Try to reshape to a square
            size = int(np.sqrt(potential_watermark.size))
            potential_watermark = potential_watermark[:size*size].reshape(size, size)
        
        # Convert to 8-bit image if not already
        if potential_watermark.dtype != np.uint8:
            potential_watermark = np.uint8(potential_watermark)
        
        # Calculate pattern metrics
        # 1. Variance - higher variance suggests a pattern rather than noise
        variance = np.var(potential_watermark) / 255.0
        
        # 2. Structure - check for structured patterns using edge detection
        edges = cv2.Canny(potential_watermark, 100, 200)
        edge_ratio = np.sum(edges > 0) / edges.size
        
        # 3. Non-randomness - check for clusters of similar values
        _, labels = cv2.connectedComponents(np.uint8(potential_watermark > 128))
        cluster_count = len(np.unique(labels))
        cluster_ratio = cluster_count / potential_watermark.size
        
        # 4. Pattern consistency - check for regular patterns (watermarks tend to have regular patterns)
        # Use FFT to detect regular patterns
        f_transform = np.fft.fft2(potential_watermark)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
        
        # Higher peaks in frequency domain indicate regular patterns
        # Normalize and take top 10% of frequencies
        if np.max(magnitude_spectrum) > 0:
            normalized_spectrum = magnitude_spectrum / np.max(magnitude_spectrum)
            pattern_strength = np.sum(normalized_spectrum > 0.9) / normalized_spectrum.size
        else:
            pattern_strength = 0
        
        # 5. Check for concentration of values - watermarks often have concentrated areas of similar values
        # Calculate histogram and check for peaks
        hist = cv2.calcHist([potential_watermark], [0], None, [256], [0, 256])
        hist_normalized = hist / np.sum(hist)
        peaks = np.sum(hist_normalized > 0.05)
        peak_ratio = peaks / 256  # Lower is better for watermarks (more concentrated values)
        
        # Combine metrics into a single score (0-1)
        # Higher variance, moderate edge_ratio, lower cluster_ratio, higher pattern_strength, lower peak_ratio
        # indicate a watermark
        score = (
            variance * 0.2 +                           # Higher variance is better
            (1.0 - abs(edge_ratio - 0.1)) * 0.2 +      # Edge ratio around 0.1 is ideal
            (1.0 - cluster_ratio) * 0.2 +              # Lower cluster ratio is better
            pattern_strength * 0.3 +                   # Higher pattern strength is better
            (1.0 - peak_ratio) * 0.1                   # Lower peak ratio is better
        )
        
        # Apply a sigmoid function to make the score more discriminative
        # This will push scores toward 0 or 1 more clearly
        score = 1.0 / (1.0 + np.exp(-10 * (score - 0.5)))
        
        # Normalize to 0-1 range
        return min(max(score, 0.0), 1.0)
        
    def create_detection_visualization(self, potential_watermark, detection_score):
        """Create a visualization of the detection result"""
        if isinstance(potential_watermark, list):
            # Handle the case where extract returns a list
            potential_watermark = potential_watermark[0]
        
        # Reshape if needed
        if len(potential_watermark.shape) == 1:
            # Try to reshape to a square
            size = int(np.sqrt(potential_watermark.size))
            potential_watermark = potential_watermark[:size*size].reshape(size, size)
        
        # Convert to 8-bit image if not already
        if potential_watermark.dtype != np.uint8:
            potential_watermark = np.uint8(potential_watermark)
        
        # Resize for better visibility
        vis_size = 400
        vis = cv2.resize(potential_watermark, (vis_size, vis_size), interpolation=cv2.INTER_NEAREST)
        
        # Apply adaptive thresholding to enhance the watermark pattern
        # This helps distinguish between actual watermark patterns and noise
        vis_enhanced = cv2.adaptiveThreshold(
            vis, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Apply color map for better visualization
        if detection_score > 0.6:  # Increase threshold for detection
            # Use a "hot" colormap for detected watermarks
            vis_color = cv2.applyColorMap(vis_enhanced, cv2.COLORMAP_HOT)
            # Add a green border to indicate detection
            border_size = 5
            vis_color = cv2.copyMakeBorder(
                vis_color, 
                border_size, border_size, border_size, border_size, 
                cv2.BORDER_CONSTANT, 
                value=(0, 255, 0)
            )
        else:
            # Use a "cool" colormap for non-detected
            vis_color = cv2.applyColorMap(vis, cv2.COLORMAP_COOL)
            # Add a red border to indicate no detection
            border_size = 5
            vis_color = cv2.copyMakeBorder(
                vis_color, 
                border_size, border_size, border_size, border_size, 
                cv2.BORDER_CONSTANT, 
                value=(0, 0, 255)
            )
        
        return vis_color


class EnhancedWatermarkingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Enhanced Digital Watermarking Application")
        self.setGeometry(100, 100, 1200, 800)
        self.setMinimumSize(800, 600)  # Set minimum window size
        self.init_ui()

    def init_ui(self):
        # Main layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Header with app info
        header = QLabel("Digital Watermarking Application")
        header.setStyleSheet("font-size: 18pt; font-weight: bold;")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        info_label = QLabel("Embed and extract watermarks using frequency domain techniques (DCT)")
        info_label.setStyleSheet("font-size: 10pt;")
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        main_layout.addWidget(header)
        main_layout.addWidget(info_label)

        # Tab widget
        tabs = QTabWidget()
        main_layout.addWidget(tabs)

        # Create tabs
        embed_tab = QWidget()
        extract_tab = QWidget()
        attack_tab = QWidget()
        evaluate_tab = QWidget()
        detect_tab = QWidget()  # New detection tab

        tabs.addTab(embed_tab, "Embed Watermark")
        tabs.addTab(extract_tab, "Extract Watermark")
        tabs.addTab(detect_tab, "Detect Watermark")  # Add the new tab
        tabs.addTab(attack_tab, "Attack Image")
        tabs.addTab(evaluate_tab, "Evaluate Robustness")

        # Set up each tab
        self.setup_embed_tab(embed_tab)
        self.setup_extract_tab(extract_tab)
        self.setup_detect_tab(detect_tab)  # Set up the new tab
        self.setup_attack_tab(attack_tab)
        self.setup_evaluate_tab(evaluate_tab)

    def setup_embed_tab(self, tab):
        layout = QVBoxLayout()
        tab.setLayout(layout)

        # Create splitter for controls and preview
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # Left side - Controls
        controls_widget = QWidget()
        controls_layout = QVBoxLayout()
        controls_widget.setLayout(controls_layout)
        splitter.addWidget(controls_widget)

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

        # Embed button and progress bar
        action_group = QGroupBox("Embed Action")
        action_layout = QVBoxLayout()
        action_group.setLayout(action_layout)
        
        self.embed_progress = QProgressBar()
        embed_btn = QPushButton("Embed Watermark")
        embed_btn.clicked.connect(self.start_embedding)
        
        action_layout.addWidget(self.embed_progress)
        action_layout.addWidget(embed_btn)

        # Add all groups to controls layout
        controls_layout.addWidget(cover_group)
        controls_layout.addWidget(signature_group)
        controls_layout.addWidget(algorithm_group)
        controls_layout.addWidget(action_group)
        controls_layout.addStretch()

        # Right side - Preview
        preview_widget = QWidget()
        preview_layout = QVBoxLayout()
        preview_widget.setLayout(preview_layout)
        splitter.addWidget(preview_widget)

        # Image preview area
        preview_group = QGroupBox("Image Preview")
        preview_inner_layout = QVBoxLayout()  # Changed to vertical layout
        preview_group.setLayout(preview_inner_layout)
        
        # Original and watermarked images in one row
        top_row_layout = QHBoxLayout()
        
        # Cover image preview
        cover_preview_container = QWidget()
        cover_preview_layout = QVBoxLayout()
        cover_preview_container.setLayout(cover_preview_layout)
        
        cover_preview_label = QLabel("Cover/Watermarked Image")
        cover_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cover_preview_layout.addWidget(cover_preview_label)
        
        self.cover_preview = QLabel()
        self.cover_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cover_preview.setMinimumSize(300, 300)
        cover_preview_layout.addWidget(self.cover_preview)
        
        top_row_layout.addWidget(cover_preview_container)
        
        # Signature image preview
        signature_preview_container = QWidget()
        signature_preview_layout = QVBoxLayout()
        signature_preview_container.setLayout(signature_preview_layout)
        
        signature_preview_label = QLabel("Signature Image")
        signature_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        signature_preview_layout.addWidget(signature_preview_label)
        
        self.signature_preview = QLabel()
        self.signature_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.signature_preview.setMinimumSize(300, 300)
        signature_preview_layout.addWidget(self.signature_preview)
        
        top_row_layout.addWidget(signature_preview_container)
        
        preview_inner_layout.addLayout(top_row_layout)
        
        # Watermark evidence in second row
        evidence_container = QWidget()
        evidence_layout = QVBoxLayout()
        evidence_container.setLayout(evidence_layout)
        
        evidence_label = QLabel("Watermark Evidence")
        evidence_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        evidence_layout.addWidget(evidence_label)
        
        # Add a slider to control evidence visibility
        slider_container = QWidget()
        slider_layout = QVBoxLayout()
        slider_container.setLayout(slider_layout)
        
        slider_header = QHBoxLayout()
        slider_label = QLabel("Evidence Visibility:")
        slider_label.setStyleSheet("font-weight: bold;")
        self.slider_value_label = QLabel("50%")
        
        slider_header.addWidget(slider_label)
        slider_header.addStretch()
        slider_header.addWidget(self.slider_value_label)
        slider_layout.addLayout(slider_header)
        
        self.evidence_slider = QSlider(Qt.Orientation.Horizontal)
        self.evidence_slider.setMinimum(0)
        self.evidence_slider.setMaximum(100)
        self.evidence_slider.setValue(50)
        self.evidence_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.evidence_slider.setTickInterval(10)
        self.evidence_slider.valueChanged.connect(self.update_evidence_visibility)
        slider_layout.addWidget(self.evidence_slider)
        
        # Add labels for slider positions
        slider_labels_layout = QHBoxLayout()
        slider_labels_layout.addWidget(QLabel("Original"))
        slider_labels_layout.addStretch()
        slider_labels_layout.addWidget(QLabel("Low"))
        slider_labels_layout.addStretch()
        slider_labels_layout.addWidget(QLabel("Medium"))
        slider_labels_layout.addStretch()
        slider_labels_layout.addWidget(QLabel("High"))
        slider_labels_layout.addStretch()
        slider_labels_layout.addWidget(QLabel("Maximum"))
        slider_layout.addLayout(slider_labels_layout)
        
        evidence_layout.addWidget(slider_container)
        
        # Create horizontal layout for evidence display
        evidence_images_layout = QHBoxLayout()
        
        # Combined evidence view (will be updated by slider)
        self.evidence_view = QLabel("After embedding, evidence will be shown here")
        self.evidence_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.evidence_view.setMinimumSize(600, 300)
        evidence_images_layout.addWidget(self.evidence_view)
        
        evidence_layout.addLayout(evidence_images_layout)
        
        preview_inner_layout.addWidget(evidence_container)
        
        preview_layout.addWidget(preview_group)

        # Initialize variables
        self.cover_image_path = ""
        self.signature_image_path = ""
        self.watermarked_image = None

        # Set splitter sizes
        splitter.setSizes([400, 800])

    def setup_extract_tab(self, tab):
        layout = QVBoxLayout()
        tab.setLayout(layout)

        # Create splitter for controls and preview
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # Left side - Controls
        controls_widget = QWidget()
        controls_layout = QVBoxLayout()
        controls_widget.setLayout(controls_layout)
        splitter.addWidget(controls_widget)

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

        # Extract button and progress bar
        extract_action_group = QGroupBox("Extract Action")
        extract_action_layout = QVBoxLayout()
        extract_action_group.setLayout(extract_action_layout)
        
        self.extract_progress = QProgressBar()
        extract_btn = QPushButton("Extract Watermark")
        extract_btn.clicked.connect(self.start_extracting)
        
        extract_action_layout.addWidget(self.extract_progress)
        extract_action_layout.addWidget(extract_btn)

        # Add all groups to controls layout
        controls_layout.addWidget(watermarked_group)
        controls_layout.addWidget(algorithm_group)
        controls_layout.addWidget(extract_action_group)
        controls_layout.addStretch()

        # Right side - Preview
        preview_widget = QWidget()
        preview_layout = QVBoxLayout()
        preview_widget.setLayout(preview_layout)
        splitter.addWidget(preview_widget)

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

        preview_layout.addWidget(extract_preview_group)

        # Initialize variables
        self.watermarked_image_path = ""
        self.extracted_signature = None

        # Set splitter sizes
        splitter.setSizes([400, 800])

    def setup_attack_tab(self, tab):
        layout = QVBoxLayout()
        tab.setLayout(layout)

        # Create splitter for controls and preview
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # Left side - Controls
        controls_widget = QWidget()
        controls_layout = QVBoxLayout()
        controls_widget.setLayout(controls_layout)
        splitter.addWidget(controls_widget)

        # Image selection
        attack_image_group = QGroupBox("Watermarked Image to Attack")
        attack_image_layout = QHBoxLayout()
        attack_image_group.setLayout(attack_image_layout)

        self.attack_image_path_label = QLabel("No file selected")
        attack_image_browse_btn = QPushButton("Browse...")
        attack_image_browse_btn.clicked.connect(lambda: self.browse_file("attack_image"))
        
        attack_image_layout.addWidget(self.attack_image_path_label)
        attack_image_layout.addWidget(attack_image_browse_btn)

        # Algorithm selection for extraction after attack
        algorithm_group = QGroupBox("Extraction Algorithm")
        algorithm_layout = QHBoxLayout()
        algorithm_group.setLayout(algorithm_layout)
        
        self.attack_dct_radio = QRadioButton("DCT")
        self.attack_dct_radio.setChecked(True)
        
        algorithm_layout.addWidget(self.attack_dct_radio)

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

        # Attack button and progress bar
        attack_action_group = QGroupBox("Attack Action")
        attack_action_layout = QVBoxLayout()
        attack_action_group.setLayout(attack_action_layout)
        
        self.attack_progress = QProgressBar()
        attack_btn = QPushButton("Apply Attack")
        attack_btn.clicked.connect(self.start_attacking)
        
        attack_action_layout.addWidget(self.attack_progress)
        attack_action_layout.addWidget(attack_btn)

        # Add all groups to controls layout
        controls_layout.addWidget(attack_image_group)
        controls_layout.addWidget(algorithm_group)
        controls_layout.addWidget(attack_type_group)
        controls_layout.addWidget(attack_action_group)
        controls_layout.addStretch()

        # Right side - Preview
        preview_widget = QWidget()
        preview_layout = QVBoxLayout()
        preview_widget.setLayout(preview_layout)
        splitter.addWidget(preview_widget)

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

        preview_layout.addWidget(attack_preview_group)

        # Initialize variables
        self.attack_image_path = ""
        self.attacked_image = None

        # Set splitter sizes
        splitter.setSizes([400, 800])

    def setup_evaluate_tab(self, tab):
        main_layout = QVBoxLayout()
        tab.setLayout(main_layout)
        
        # Header with description
        header = QLabel("Watermark Robustness Evaluation")
        header.setStyleSheet("font-size: 16pt; font-weight: bold; margin-bottom: 10px;")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        description = QLabel("Analyze how well your watermark survives against different types of attacks")
        description.setStyleSheet("font-size: 10pt; color: #555555; margin-bottom: 15px;")
        description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        main_layout.addWidget(header)
        main_layout.addWidget(description)
        
        # Main content splitter (horizontal layout)
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(main_splitter, 1)  # Give it stretch factor
        
        # === Left panel: Controls ===
        controls_widget = QWidget()
        controls_layout = QVBoxLayout()
        controls_widget.setLayout(controls_layout)
        controls_widget.setMinimumWidth(300)
        
        # Step 1: Select Image
        step1_group = QGroupBox("Step 1: Select Watermarked Image")
        step1_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        step1_layout = QVBoxLayout()
        step1_group.setLayout(step1_layout)
        
        image_layout = QHBoxLayout()
        self.eval_image_path_label = QLabel("No file selected")
        self.eval_image_path_label.setStyleSheet("background-color: #f5f5f5; padding: 5px; border-radius: 3px;")
        self.eval_image_preview_small = QLabel()
        self.eval_image_preview_small.setFixedSize(80, 80)
        self.eval_image_preview_small.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.eval_image_preview_small.setStyleSheet("background-color: #e0e0e0; border-radius: 5px;")
        
        image_browse_btn = QPushButton("Browse...")
        image_browse_btn.setStyleSheet("background-color: #3a86ff; color: white; font-weight: bold; padding: 5px 15px;")
        image_browse_btn.clicked.connect(lambda: self.browse_file("evaluate"))
        
        image_layout.addWidget(self.eval_image_path_label, 1)
        image_layout.addWidget(image_browse_btn)
        
        step1_layout.addLayout(image_layout)
        step1_layout.addWidget(self.eval_image_preview_small, 0, Qt.AlignmentFlag.AlignCenter)
        
        # Step 2: Algorithm Selection
        step2_group = QGroupBox("Step 2: Select Algorithm")
        step2_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        step2_layout = QVBoxLayout()
        step2_group.setLayout(step2_layout)
        
        self.eval_dct_radio = QRadioButton("DCT (Discrete Cosine Transform)")
        self.eval_dct_radio.setChecked(True)
        self.eval_dct_radio.setStyleSheet("padding: 5px;")
        
        algorithm_description = QLabel("DCT embeds watermarks in the frequency domain of the image")
        algorithm_description.setStyleSheet("font-style: italic; color: #555555; padding-left: 20px;")
        algorithm_description.setWordWrap(True)
        
        step2_layout.addWidget(self.eval_dct_radio)
        step2_layout.addWidget(algorithm_description)
        
        # Step 3: Attack Selection
        step3_group = QGroupBox("Step 3: Select Attack Type")
        step3_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        step3_layout = QVBoxLayout()
        step3_group.setLayout(step3_layout)
        
        # Organize attacks by categories in a scrollable area
        attacks_scroll = QScrollArea()
        attacks_scroll.setWidgetResizable(True)
        attacks_scroll.setStyleSheet("QScrollArea { border: none; }")
        
        attacks_container = QWidget()
        attacks_layout = QVBoxLayout()
        attacks_container.setLayout(attacks_layout)
        
        # Create a single button group for all attack types
        from PyQt5.QtWidgets import QButtonGroup
        self.attack_button_group = QButtonGroup()
        
        # Create attack categories
        self.eval_attacks = {}
        
        # Geometric attacks
        geometric_group = QGroupBox("Geometric Attacks")
        geometric_group.setStyleSheet("QGroupBox { font-weight: bold; color: #e63946; }")
        geometric_layout = QVBoxLayout()
        geometric_group.setLayout(geometric_layout)
        
        self.eval_attacks["rotate90"] = QRadioButton("Rotate 90°")
        self.eval_attacks["rotate180"] = QRadioButton("Rotate 180°")
        self.eval_attacks["chop5"] = QRadioButton("Crop 5%")
        self.eval_attacks["chop10"] = QRadioButton("Crop 10%")
        self.eval_attacks["chop30"] = QRadioButton("Crop 30%")
        self.eval_attacks["largersize"] = QRadioButton("Enlarge")
        self.eval_attacks["smallersize"] = QRadioButton("Shrink")
        
        for attack in ["rotate90", "rotate180", "chop5", "chop10", "chop30", "largersize", "smallersize"]:
            geometric_layout.addWidget(self.eval_attacks[attack])
            # Add each radio button to the button group
            self.attack_button_group.addButton(self.eval_attacks[attack])
        
        # Noise attacks
        noise_group = QGroupBox("Noise & Filter Attacks")
        noise_group.setStyleSheet("QGroupBox { font-weight: bold; color: #118ab2; }")
        noise_layout = QVBoxLayout()
        noise_group.setLayout(noise_layout)
        
        self.eval_attacks["blur"] = QRadioButton("Gaussian Blur")
        self.eval_attacks["saltnoise"] = QRadioButton("Salt & Pepper Noise")
        self.eval_attacks["randline"] = QRadioButton("Random Lines")
        self.eval_attacks["cover"] = QRadioButton("Cover Regions")
        
        for attack in ["blur", "saltnoise", "randline", "cover"]:
            noise_layout.addWidget(self.eval_attacks[attack])
            # Add each radio button to the button group
            self.attack_button_group.addButton(self.eval_attacks[attack])
        
        # Color/intensity attacks
        color_group = QGroupBox("Color & Intensity Attacks")
        color_group.setStyleSheet("QGroupBox { font-weight: bold; color: #06d6a0; }")
        color_layout = QVBoxLayout()
        color_group.setLayout(color_layout)
        
        self.eval_attacks["gray"] = QRadioButton("Grayscale Conversion")
        self.eval_attacks["brighter10"] = QRadioButton("Increase Brightness 10%")
        self.eval_attacks["darker10"] = QRadioButton("Decrease Brightness 10%")
        
        for attack in ["gray", "brighter10", "darker10"]:
            color_layout.addWidget(self.eval_attacks[attack])
            # Add each radio button to the button group
            self.attack_button_group.addButton(self.eval_attacks[attack])
        
        # Add all categories to the attacks layout
        attacks_layout.addWidget(geometric_group)
        attacks_layout.addWidget(noise_group)
        attacks_layout.addWidget(color_group)
        attacks_layout.addStretch()
        
        # Set default selection
        self.eval_attacks["blur"].setChecked(True)
        
        attacks_scroll.setWidget(attacks_container)
        step3_layout.addWidget(attacks_scroll)
        
        # Buttons: Run Evaluation and Reset
        eval_btn = QPushButton("▶ Run Evaluation")
        eval_btn.setMinimumHeight(50)
        eval_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 14pt;
                font-weight: bold;
                border-radius: 5px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3e8e41;
            }
        """)
        eval_btn.clicked.connect(self.run_evaluation)
        
        # Add reset button
        reset_btn = QPushButton("⟲ Reset")
        reset_btn.setMinimumHeight(40)
        reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-size: 12pt;
                font-weight: bold;
                border-radius: 5px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
            QPushButton:pressed {
                background-color: #b71c1c;
            }
        """)
        reset_btn.clicked.connect(self.reset_evaluation)
        
        # Create button layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(eval_btn, 2)  # Give more space to the evaluation button
        button_layout.addWidget(reset_btn, 1)
        
        # Add all steps to controls layout
        controls_layout.addWidget(step1_group)
        controls_layout.addWidget(step2_group)
        controls_layout.addWidget(step3_group)
        controls_layout.addLayout(button_layout)
        
        # === Right panel: Results ===
        results_widget = QWidget()
        results_layout = QVBoxLayout()
        results_widget.setLayout(results_layout)
        
        # Results header
        results_header = QLabel("Evaluation Results")
        results_header.setStyleSheet("font-size: 14pt; font-weight: bold;")
        results_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Results container (vertical split between report and visualization)
        results_splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Top part: HTML Report in scrollable area
        report_container = QWidget()
        report_layout = QVBoxLayout()
        report_container.setLayout(report_layout)
        
        report_label = QLabel("Watermark Analysis Report")
        report_label.setStyleSheet("font-weight: bold; color: #444;")
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("""
            QScrollArea {
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 5px;
            }
        """)
        
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout()
        scroll_content.setLayout(scroll_layout)
        
        self.eval_results = QLabel("Run evaluation to see detailed analysis")
        self.eval_results.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.eval_results.setWordWrap(True)
        self.eval_results.setTextFormat(Qt.TextFormat.RichText)
        self.eval_results.setStyleSheet("""
            QLabel {
                padding: 10px;
                background-color: white;
                color: #333;
            }
        """)
        scroll_layout.addWidget(self.eval_results)
        scroll_layout.addStretch()
        
        scroll_area.setWidget(scroll_content)
        
        report_layout.addWidget(report_label)
        report_layout.addWidget(scroll_area)
        
        # Bottom part: Visual comparison
        visual_container = QWidget()
        visual_layout = QVBoxLayout()
        visual_container.setLayout(visual_layout)
        
        visual_label = QLabel("Visual Comparison")
        visual_label.setStyleSheet("font-weight: bold; color: #444;")
        
        self.eval_image_preview = QLabel("Visual comparison will appear here after evaluation")
        self.eval_image_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.eval_image_preview.setMinimumHeight(200)
        self.eval_image_preview.setStyleSheet("""
            QLabel {
                background-color: #f8f9fa;
                border: 1px dashed #ccc;
                border-radius: 5px;
                color: #666;
            }
        """)
        
        visual_layout.addWidget(visual_label)
        visual_layout.addWidget(self.eval_image_preview)
        
        # Add to results splitter
        results_splitter.addWidget(report_container)
        results_splitter.addWidget(visual_container)
        results_splitter.setSizes([300, 300])  # Equal initial sizes
        
        # Add header and results splitter to results layout
        results_layout.addWidget(results_header)
        results_layout.addWidget(results_splitter, 1)  # Give it stretch factor
        
        # Add both panels to main splitter
        main_splitter.addWidget(controls_widget)
        main_splitter.addWidget(results_widget)
        main_splitter.setSizes([400, 800])  # 1:2 ratio for controls vs results

    def setup_detect_tab(self, tab):
        layout = QVBoxLayout()
        tab.setLayout(layout)

        # Create splitter for controls and preview
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # Left side - Controls
        controls_widget = QWidget()
        controls_layout = QVBoxLayout()
        controls_widget.setLayout(controls_layout)
        splitter.addWidget(controls_widget)

        # Image selection
        image_group = QGroupBox("Image to Check")
        image_layout = QHBoxLayout()
        image_group.setLayout(image_layout)

        self.detect_image_path_label = QLabel("No file selected")
        image_browse_btn = QPushButton("Browse...")
        image_browse_btn.clicked.connect(lambda: self.browse_file("detect_image"))
        
        image_layout.addWidget(self.detect_image_path_label)
        image_layout.addWidget(image_browse_btn)
        
        # Sample signature selection (new)
        sample_group = QGroupBox("Sample Signature (Optional)")
        sample_layout = QVBoxLayout()
        sample_group.setLayout(sample_layout)
        
        # Add checkbox to enable/disable comparison mode
        self.compare_checkbox = QRadioButton("Compare with sample signature")
        self.detect_only_checkbox = QRadioButton("Just detect any watermark")
        self.detect_only_checkbox.setChecked(True)  # Default to detection only
        
        sample_layout.addWidget(self.detect_only_checkbox)
        sample_layout.addWidget(self.compare_checkbox)
        
        # Sample signature file selection
        sample_file_layout = QHBoxLayout()
        self.sample_signature_path_label = QLabel("No sample selected")
        sample_browse_btn = QPushButton("Browse...")
        sample_browse_btn.clicked.connect(lambda: self.browse_file("sample_signature"))
        
        sample_file_layout.addWidget(self.sample_signature_path_label)
        sample_file_layout.addWidget(sample_browse_btn)
        sample_layout.addLayout(sample_file_layout)
        
        # Sample signature preview
        self.sample_signature_preview = QLabel("Sample Signature Preview")
        self.sample_signature_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sample_signature_preview.setMinimumSize(200, 100)
        self.sample_signature_preview.setMaximumSize(200, 100)
        sample_layout.addWidget(self.sample_signature_preview)

        # Algorithm selection
        algorithm_group = QGroupBox("Detection Algorithm")
        algorithm_layout = QHBoxLayout()
        algorithm_group.setLayout(algorithm_layout)
        
        self.detect_dct_radio = QRadioButton("DCT")
        self.detect_dct_radio.setChecked(True)
        
        algorithm_layout.addWidget(self.detect_dct_radio)

        # Detection sensitivity slider
        sensitivity_group = QGroupBox("Detection Sensitivity")
        sensitivity_layout = QVBoxLayout()
        sensitivity_group.setLayout(sensitivity_layout)
        
        slider_header = QHBoxLayout()
        sensitivity_label = QLabel("Sensitivity Level:")
        sensitivity_label.setStyleSheet("font-weight: bold;")
        self.sensitivity_value_label = QLabel("50%")
        
        slider_header.addWidget(sensitivity_label)
        slider_header.addStretch()
        slider_header.addWidget(self.sensitivity_value_label)
        sensitivity_layout.addLayout(slider_header)
        
        self.sensitivity_slider = QSlider(Qt.Orientation.Horizontal)
        self.sensitivity_slider.setMinimum(0)
        self.sensitivity_slider.setMaximum(100)
        self.sensitivity_slider.setValue(50)
        self.sensitivity_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.sensitivity_slider.setTickInterval(10)
        self.sensitivity_slider.valueChanged.connect(self.update_sensitivity_value)
        sensitivity_layout.addWidget(self.sensitivity_slider)
        
        # Add labels for slider positions
        slider_labels_layout = QHBoxLayout()
        slider_labels_layout.addWidget(QLabel("Low"))
        slider_labels_layout.addStretch()
        slider_labels_layout.addWidget(QLabel("Medium"))
        slider_labels_layout.addStretch()
        slider_labels_layout.addWidget(QLabel("High"))
        sensitivity_layout.addLayout(slider_labels_layout)

        # Detect button and progress bar
        detect_action_group = QGroupBox("Detection Action")
        detect_action_layout = QVBoxLayout()
        detect_action_group.setLayout(detect_action_layout)
        
        self.detect_progress = QProgressBar()
        detect_btn = QPushButton("Detect Watermark")
        detect_btn.clicked.connect(self.start_detecting)
        
        detect_action_layout.addWidget(self.detect_progress)
        detect_action_layout.addWidget(detect_btn)

        # Add all groups to controls layout
        controls_layout.addWidget(image_group)
        controls_layout.addWidget(sample_group)
        controls_layout.addWidget(algorithm_group)
        controls_layout.addWidget(sensitivity_group)
        controls_layout.addWidget(detect_action_group)
        controls_layout.addStretch()

        # Right side - Preview
        preview_widget = QWidget()
        preview_layout = QVBoxLayout()
        preview_widget.setLayout(preview_layout)
        splitter.addWidget(preview_widget)

        # Image preview area
        detect_preview_group = QGroupBox("Detection Results")
        detect_preview_layout = QVBoxLayout()
        detect_preview_group.setLayout(detect_preview_layout)
        
        # Original image preview
        self.detect_image_preview = QLabel("Image to Check")
        self.detect_image_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detect_image_preview.setMinimumSize(400, 300)
        detect_preview_layout.addWidget(self.detect_image_preview)
        
        # Detection result display
        self.detect_result_group = QGroupBox("Watermark Detection Result")
        detect_result_layout = QVBoxLayout()
        self.detect_result_group.setLayout(detect_result_layout)
        
        self.detect_result_label = QLabel("No detection performed yet")
        self.detect_result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detect_result_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
        detect_result_layout.addWidget(self.detect_result_label)
        
        # Confidence meter
        confidence_layout = QHBoxLayout()
        confidence_label = QLabel("Confidence:")
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setMinimum(0)
        self.confidence_bar.setMaximum(100)
        self.confidence_bar.setValue(0)
        
        confidence_layout.addWidget(confidence_label)
        confidence_layout.addWidget(self.confidence_bar)
        detect_result_layout.addLayout(confidence_layout)
        
        # Detection details
        self.detection_details = QLabel("No detection details available")
        self.detection_details.setWordWrap(True)
        detect_result_layout.addWidget(self.detection_details)
        
        detect_preview_layout.addWidget(self.detect_result_group)
        preview_layout.addWidget(detect_preview_group)

        # Initialize variables
        self.detect_image_path = ""
        self.detection_sensitivity = 50  # Default value

        # Set splitter sizes
        splitter.setSizes([400, 800])
        
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
            
        elif file_type == "evaluate":
            self.eval_image_path = file_path
            self.eval_image_path_label.setText(os.path.basename(file_path))
            # Update the small preview image in evaluation tab
            if hasattr(self, 'eval_image_preview_small'):
                self.display_image(file_path, self.eval_image_preview_small)
            
        elif file_type == "detect_image":
            self.detect_image_path = file_path
            self.detect_image_path_label.setText(os.path.basename(file_path))
            self.display_image(file_path, self.detect_image_preview)
            
        elif file_type == "sample_signature":
            self.sample_signature_path = file_path
            self.sample_signature_path_label.setText(os.path.basename(file_path))
            self.display_image(file_path, self.sample_signature_preview)
            # Auto-select the comparison mode when a sample is loaded
            self.compare_checkbox.setChecked(True)

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

    def embedding_finished(self, result_img, message, original_img):
        self.watermarked_image = result_img
        self.display_image(result_img, self.cover_preview)
        
        # Create difference image to show watermark evidence
        if original_img is not None:
            # Store original and watermarked images for slider functionality
            self.original_image = original_img
            
            # Calculate the absolute difference between original and watermarked images
            diff_img = cv2.absdiff(original_img, result_img)
            
            # Save the original difference image
            diff_path = os.path.join("output", "watermark_evidence_original.png")
            cv2.imwrite(diff_path, diff_img)
            
            # Store the difference image for slider functionality
            self.diff_image = diff_img
            
            # Enhance the difference for better visibility with different alpha values
            self.enhanced_diff_low = cv2.convertScaleAbs(diff_img, alpha=5.0)
            self.enhanced_diff_medium = cv2.convertScaleAbs(diff_img, alpha=20.0)
            self.enhanced_diff_high = cv2.convertScaleAbs(diff_img, alpha=50.0)
            
            # Create color mapped versions
            self.enhanced_color_low = cv2.applyColorMap(self.enhanced_diff_low, cv2.COLORMAP_VIRIDIS)
            self.enhanced_color_medium = cv2.applyColorMap(self.enhanced_diff_medium, cv2.COLORMAP_VIRIDIS)
            self.enhanced_color_high = cv2.applyColorMap(self.enhanced_diff_high, cv2.COLORMAP_VIRIDIS)
            
            # Save the enhanced color version
            enhanced_color_path = os.path.join("output", "watermark_evidence_enhanced.png")
            cv2.imwrite(enhanced_color_path, self.enhanced_color_medium)
            
            # Display the signature in the signature preview
            if hasattr(self, 'signature_image_path') and self.signature_image_path:
                self.display_image(self.signature_image_path, self.signature_preview)
            
            # Set initial evidence display
            self.update_evidence_visibility()
            
            # Update message to include evidence information
            message += "\nWatermark evidence has been generated. Use the slider to adjust visibility."
        
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

    def extracting_finished(self, result_img, message, original_img):
        self.extracted_signature = result_img
        
        # Enhance the extracted signature for better visibility
        # First, make sure it's binary (0 or 255)
        binary_sig = np.where(result_img > 128, 255, 0).astype(np.uint8)
        
        # Create a more visible version with larger dots for the watermark
        enhanced_sig = np.zeros((binary_sig.shape[0]*2, binary_sig.shape[1]*2), dtype=np.uint8)
        for i in range(binary_sig.shape[0]):
            for j in range(binary_sig.shape[1]):
                if binary_sig[i, j] == 255:
                    # Create a 2x2 white dot for each white pixel
                    enhanced_sig[i*2:i*2+2, j*2:j*2+2] = 255
        
        # Save the enhanced signature
        enhanced_sig_path = os.path.join("output", "enhanced_extracted_signature.jpg")
        cv2.imwrite(enhanced_sig_path, enhanced_sig)
        
        # Display the enhanced signature
        self.display_image(enhanced_sig, self.extracted_preview)
        
        # Update message
        message += "\nExtracted signature has been enhanced for better visibility."
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

    def attacking_finished(self, result_img, message, original_img):
        self.attacked_image = result_img
        self.display_image(result_img, self.attack_output_preview)
        QMessageBox.information(self, "Success", message)

    def run_evaluation(self):
        if not hasattr(self, 'eval_image_path'):
            QMessageBox.warning(self, "Input Error", "Please select a watermarked image to evaluate")
            return

        # Get selected attacks
        selected_attack = None
        for attack_name, radio_btn in self.eval_attacks.items():
            if radio_btn.isChecked():
                selected_attack = attack_name
                break

        if not selected_attack:
            QMessageBox.warning(self, "Input Error", "Please select at least one attack type")
            return

        # Start evaluation
        self.start_evaluation_process(selected_attack)

    def start_evaluation_process(self, attack_type):
        """Start the evaluation process for the selected attack"""
        results_dir = os.path.join("output", "evaluation_results")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            
        # First, create attacked image
        output_path = os.path.join(results_dir, f"attacked_{attack_type}.png")
        
        # Initialize worker for attack
        self.eval_worker = WatermarkWorker(
            mode="attack",
            algorithm="DCT",
            cover_path=self.eval_image_path,
            signature_path=None,
            output_path=output_path
        )
        self.eval_worker.set_attack_type(attack_type)
        
        # Connect signals
        self.eval_worker.progress.connect(lambda v: self.eval_results.setText(f"Processing attack ({v}%)..."))
        self.eval_worker.finished.connect(lambda img, msg, orig: self.attack_evaluation_finished(img, msg, orig, attack_type))
        self.eval_worker.error.connect(self.show_error)
        
        # Start worker
        self.eval_worker.start()

    def attack_evaluation_finished(self, attacked_img, message, original_img, attack_type):
        """Called when attack phase of evaluation is complete"""
        results_dir = os.path.join("output", "evaluation_results")
        
        # Now extract watermark from attacked image
        output_path = os.path.join(results_dir, f"extracted_from_{attack_type}.png")
        
        # Initialize worker for extraction
        self.eval_extract_worker = WatermarkWorker(
            mode="extract",
            algorithm="DCT",
            cover_path=os.path.join(results_dir, f"attacked_{attack_type}.png"),
            signature_path=None,
            output_path=output_path
        )
        
        # Connect signals
        self.eval_extract_worker.progress.connect(lambda v: self.eval_results.setText(f"Extracting watermark ({v}%)..."))
        self.eval_extract_worker.finished.connect(
            lambda img, msg, orig: self.extraction_evaluation_finished(img, msg, orig, attack_type, attacked_img)
        )
        self.eval_extract_worker.error.connect(self.show_error)
        
        # Start worker
        self.eval_extract_worker.start()

    def extraction_evaluation_finished(self, extracted_watermark, message, original_img, attack_type, attacked_img):
        """Called when extraction phase of evaluation is complete"""
        try:
            # Use our new WatermarkEvaluator class
            evaluator = WatermarkEvaluator()
            
            # Generate evaluation report
            report = evaluator.generate_evaluation_report(attack_type, attacked_img, extracted_watermark)
            
            # Create visual comparison
            visual_path = evaluator.create_visual_comparison(attacked_img, extracted_watermark)
            
            # Generate HTML report
            html_report = evaluator.generate_html_report(report, visual_path)
            
            # Update the results label with the HTML report
            self.eval_results.setText(html_report)
            self.eval_results.setTextFormat(Qt.TextFormat.RichText)
            
            # Display the comparison image if available
            if visual_path and os.path.exists(visual_path):
                self.display_evaluation_image(visual_path)
            
            # Show a success message
            QMessageBox.information(self, "Evaluation Complete", 
                                   f"Evaluation for {attack_type} attack completed. Results saved to {evaluator.results_dir}")
        except Exception as e:
            QMessageBox.critical(self, "Evaluation Error", f"Error during evaluation: {str(e)}")

    def display_evaluation_image(self, image_path):
        """Display the evaluation comparison image in the GUI"""
        try:
            # Load image using QPixmap
            pixmap = QPixmap(image_path)
            
            # Scale pixmap while preserving aspect ratio
            pixmap = pixmap.scaled(
                self.eval_image_preview.width(),
                self.eval_image_preview.height(),
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            )
            
            # Set the pixmap to the image preview label
            self.eval_image_preview.setPixmap(pixmap)
        except Exception as e:
            self.eval_image_preview.setText(f"Error displaying image: {str(e)}")
            print(f"Error displaying evaluation image: {str(e)}")

    def show_error(self, error_message):
        QMessageBox.critical(self, "Error", error_message)

    def update_evidence_visibility(self):
        # This method updates the evidence display based on the slider value
        if not hasattr(self, 'watermarked_image') or not hasattr(self, 'original_image'):
            return
            
        slider_value = self.evidence_slider.value()
        
        # Update the slider value label
        self.slider_value_label.setText(f"{slider_value}%")
        
        if slider_value <= 20:
            # Show mostly the watermarked image with slight evidence
            alpha = slider_value / 20.0  # 0 to 1
            beta = 1.0 - alpha
            blend = cv2.addWeighted(self.watermarked_image, beta, self.enhanced_color_low, alpha, 0)
            self.display_image(blend, self.evidence_view)
        elif slider_value <= 50:
            # Show medium enhanced evidence
            alpha = (slider_value - 20) / 30.0  # 0 to 1
            beta = 1.0 - alpha
            blend = cv2.addWeighted(self.enhanced_color_low, beta, self.enhanced_color_medium, alpha, 0)
            self.display_image(blend, self.evidence_view)
        elif slider_value <= 80:
            # Show high enhanced evidence
            alpha = (slider_value - 50) / 30.0  # 0 to 1
            beta = 1.0 - alpha
            blend = cv2.addWeighted(self.enhanced_color_medium, beta, self.enhanced_color_high, alpha, 0)
            self.display_image(blend, self.evidence_view)
        else:
            # Show maximum evidence with just the high contrast difference
            self.display_image(self.enhanced_color_high, self.evidence_view)

    def start_detecting(self):
        if not self.detect_image_path:
            QMessageBox.warning(self, "Input Error", "Please select an image to detect")
            return
            
        sensitivity = self.sensitivity_slider.value() / 100.0  # Convert to 0-1 range
        output_path = os.path.join("output", f"detected_result_{os.path.basename(self.detect_image_path)}")
        
        # Initialize worker thread
        self.detect_worker = WatermarkWorker(
            mode="detect",
            algorithm="DCT",
            cover_path=self.detect_image_path,
            signature_path=None,
            output_path=output_path
        )
        self.detect_worker.set_sensitivity(sensitivity)
        
        # Set comparison mode if enabled and sample signature is available
        comparison_mode = self.compare_checkbox.isChecked()
        if comparison_mode and hasattr(self, 'sample_signature_path'):
            if not self.sample_signature_path:
                QMessageBox.warning(self, "Input Error", "Please select a sample signature for comparison")
                return
            self.detect_worker.set_comparison_mode(True, self.sample_signature_path)
            
            # Set a stricter threshold for signature comparison
            # This makes the comparison more demanding when comparing specific signatures
            adjusted_sensitivity = min(0.9, sensitivity * 1.2)  # Increase sensitivity but cap at 0.9
            self.detect_worker.set_sensitivity(adjusted_sensitivity)
        else:
            self.detect_worker.set_comparison_mode(False)
        
        # Connect signals
        self.detect_worker.progress.connect(self.detect_progress.setValue)
        self.detect_worker.finished.connect(self.detecting_finished)
        self.detect_worker.error.connect(self.show_error)
        
        # Start worker
        self.detect_progress.setValue(0)
        self.detect_worker.start()

    def detecting_finished(self, result_img, message, original_img):
        self.detected_image = result_img
        self.display_image(result_img, self.detect_image_preview)
        
        # Extract confidence/similarity value from the message
        if "Similarity:" in message:
            # This is a comparison result
            similarity_str = message.split("Similarity: ")[1].split("%")[0]
            try:
                confidence = int(similarity_str)
            except ValueError:
                confidence = 0
                
            # Update detection result label
            self.detect_result_label.setText(message)
            
            # Update confidence bar
            self.confidence_bar.setValue(confidence)
            
            # Update detection details with more information
            if "MATCH FOUND" in message:
                self.detection_details.setText(
                    f"The image contains the specific watermark.\n"
                    f"Similarity with sample: {confidence}%\n"
                    f"Sensitivity setting: {self.sensitivity_slider.value()}%"
                )
                # Set a green color for positive match
                self.detect_result_label.setStyleSheet("font-size: 14pt; font-weight: bold; color: green;")
            else:
                self.detection_details.setText(
                    f"The image does not contain the specific watermark.\n"
                    f"Similarity with sample: {confidence}%\n"
                    f"Sensitivity setting: {self.sensitivity_slider.value()}%"
                )
                # Set a red color for no match
                self.detect_result_label.setStyleSheet("font-size: 14pt; font-weight: bold; color: red;")
        else:
            # This is a standard detection result (existing code)
            confidence_str = message.split("with ")[1].split("%")[0] if "with" in message else "0"
            try:
                confidence = int(confidence_str)
            except ValueError:
                confidence = 0
                
            # Update detection result label
            self.detect_result_label.setText(message)
            
            # Update confidence bar
            self.confidence_bar.setValue(confidence)
            
            # Update detection details with more information
            if "DETECTED" in message:
                self.detection_details.setText(
                    f"Detected watermark with {confidence}% confidence.\n"
                    f"The image appears to contain an embedded watermark pattern.\n"
                    f"Sensitivity setting: {self.sensitivity_slider.value()}%"
                )
                # Set a green color for positive detection
                self.detect_result_label.setStyleSheet("font-size: 14pt; font-weight: bold; color: green;")
            else:
                self.detection_details.setText(
                    f"No watermark detected (confidence: {confidence}%).\n"
                    f"The image does not appear to contain an embedded watermark pattern.\n"
                    f"Sensitivity setting: {self.sensitivity_slider.value()}%"
                )
                # Set a red color for negative detection
                self.detect_result_label.setStyleSheet("font-size: 14pt; font-weight: bold; color: red;")
        
        QMessageBox.information(self, "Detection Complete", message)

    def update_sensitivity_value(self):
        # This method updates the sensitivity value label based on the slider value
        slider_value = self.sensitivity_slider.value()
        self.sensitivity_value_label.setText(f"{slider_value}%")

    # Add reset evaluation method
    def reset_evaluation(self):
        """Reset the evaluation tab to its initial state"""
        # Clear selected image
        if hasattr(self, 'eval_image_path'):
            delattr(self, 'eval_image_path')
        self.eval_image_path_label.setText("No file selected")
        
        # Clear the small preview
        if hasattr(self, 'eval_image_preview_small'):
            self.eval_image_preview_small.clear()
            self.eval_image_preview_small.setText("")
        
        # Clear the results
        self.eval_results.setText("Run evaluation to see detailed analysis")
        
        # Clear the image preview
        self.eval_image_preview.clear()
        self.eval_image_preview.setText("Visual comparison will appear here after evaluation")
        
        # Reset attack selection - ensure only Gaussian Blur is selected
        for attack, radio_btn in self.eval_attacks.items():
            # First uncheck all
            radio_btn.setChecked(False)
        
        # Then only check the blur option
        if "blur" in self.eval_attacks:
            self.eval_attacks["blur"].setChecked(True)
        
        # Show confirmation
        QMessageBox.information(self, "Reset Complete", "Evaluation panel has been reset to initial state")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look across platforms
    window = EnhancedWatermarkingApp()
    window.show()
    sys.exit(app.exec_()) 