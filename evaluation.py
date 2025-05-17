import os
import cv2
import numpy as np
from PyQt5.QtCore import Qt

class WatermarkEvaluator:
    """Class for evaluating watermark robustness against various attacks"""
    
    def __init__(self):
        """Initialize the evaluator"""
        # Create results directory if it doesn't exist
        self.results_dir = os.path.join("output", "evaluation_results")
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
    
    def generate_evaluation_report(self, attack_type, attacked_img, extracted_watermark):
        """Generate a detailed evaluation report"""
        report = {
            'attack_type': attack_type,
            'image_quality_metrics': self.calculate_image_quality_metrics(attacked_img),
            'watermark_metrics': self.calculate_watermark_metrics(extracted_watermark)
        }
        return report
    
    def calculate_image_quality_metrics(self, attacked_img):
        """Calculate quality metrics for the attacked image"""
        metrics = {}
        
        # Calculate standard deviation (measure of image contrast)
        std_dev = np.std(attacked_img)
        metrics['contrast'] = std_dev
        
        # Calculate average brightness
        avg_brightness = np.mean(attacked_img)
        metrics['brightness'] = avg_brightness
        
        # Calculate image entropy (measure of information content)
        try:
            hist, _ = np.histogram(attacked_img.flatten(), bins=256, density=True)
            hist = hist[hist > 0]  # Remove zero probabilities
            entropy = -np.sum(hist * np.log2(hist))
        except:
            entropy = 0
        metrics['entropy'] = entropy
        
        return metrics
    
    def calculate_watermark_metrics(self, extracted_watermark):
        """Calculate metrics for the extracted watermark"""
        metrics = {}
        
        # Handle different watermark formats
        try:
            # Convert to 8-bit format if needed
            if extracted_watermark.dtype != np.uint8:
                if extracted_watermark.dtype == np.float32 or extracted_watermark.dtype == np.float64:
                    # Scale float values to 0-255 range
                    normalized = cv2.normalize(extracted_watermark, None, 0, 255, cv2.NORM_MINMAX)
                    extracted_watermark = normalized.astype(np.uint8)
                elif extracted_watermark.dtype == np.int32:
                    # Handle 32-bit integers by taking absolute values and scaling
                    abs_values = np.abs(extracted_watermark)
                    max_val = np.max(abs_values) if np.max(abs_values) > 0 else 1
                    scaled = (abs_values * 255 / max_val).astype(np.uint8)
                    extracted_watermark = scaled
                else:
                    # Generic conversion attempt
                    extracted_watermark = extracted_watermark.astype(np.uint8)
            
            # Ensure the watermark is grayscale
            if len(extracted_watermark.shape) > 2:
                extracted_watermark = cv2.cvtColor(extracted_watermark, cv2.COLOR_BGR2GRAY)
            
            # Calculate average intensity of the watermark
            avg_intensity = np.mean(extracted_watermark)
            metrics['intensity'] = avg_intensity
            
            # Calculate watermark contrast
            contrast = np.std(extracted_watermark)
            metrics['contrast'] = contrast
            
            # Calculate number of distinct regions (measure of watermark integrity)
            # Apply thresholding first to create a binary image
            _, binary_watermark = cv2.threshold(extracted_watermark, 127, 255, cv2.THRESH_BINARY)
            _, labels = cv2.connectedComponents(binary_watermark)
            metrics['distinct_regions'] = len(np.unique(labels)) - 1  # Subtract 1 to exclude background
        
        except Exception as e:
            print(f"Error calculating watermark metrics: {str(e)}")
            metrics['intensity'] = 0
            metrics['contrast'] = 0
            metrics['distinct_regions'] = 0
        
        return metrics
    
    def evaluate_watermark_quality(self, metrics):
        """Evaluate overall watermark quality based on metrics"""
        # Define thresholds for good quality
        intensity_threshold = 50
        contrast_threshold = 30
        regions_threshold = 5
        
        # Calculate score based on metrics
        score = 0
        if metrics['intensity'] > intensity_threshold:
            score += 1
        if metrics['contrast'] > contrast_threshold:
            score += 1
        if metrics['distinct_regions'] > regions_threshold:
            score += 1
        
        # Return quality assessment
        if score >= 3:
            return "Excellent - Watermark survived the attack well"
        elif score == 2:
            return "Good - Watermark is still detectable but somewhat degraded"
        elif score == 1:
            return "Poor - Watermark is severely degraded"
        else:
            return "Failed - Watermark is not recoverable"
    
    def create_visual_comparison(self, attacked_img, extracted_watermark):
        """Create a visual comparison of attacked image and extracted watermark"""
        try:
            # Ensure the watermark is in the right format for display
            if extracted_watermark.dtype != np.uint8:
                # Normalize to 0-255 range
                normalized = cv2.normalize(extracted_watermark, None, 0, 255, cv2.NORM_MINMAX)
                display_watermark = normalized.astype(np.uint8)
            else:
                display_watermark = extracted_watermark.copy()
            
            # Handle special case for grayscale attacked images (from grayscale conversion attack)
            attacked_display = attacked_img.copy()
            if len(attacked_display.shape) == 2:  # Grayscale image
                attacked_display = cv2.cvtColor(attacked_display, cv2.COLOR_GRAY2BGR)
            
            # If the watermark is grayscale, convert to BGR for consistent display
            if len(display_watermark.shape) == 2:
                display_watermark = cv2.cvtColor(display_watermark, cv2.COLOR_GRAY2BGR)
            
            # Resize watermark to a reasonable size for display
            wm_height, wm_width = display_watermark.shape[:2]
            img_height, img_width = attacked_display.shape[:2]
            
            # Scale watermark to match image height
            scale_factor = img_height / wm_height
            new_width = int(wm_width * scale_factor)
            resized_watermark = cv2.resize(display_watermark, (new_width, img_height))
            
            # Create side-by-side comparison
            comparison_image = np.hstack((attacked_display, resized_watermark))
            
            # Add labels to the images
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(comparison_image, "Attacked Image", (10, 30), font, 0.7, (255, 255, 255), 2)
            cv2.putText(comparison_image, "Extracted Watermark", (img_width + 10, 30), font, 0.7, (255, 255, 255), 2)
            
            # Save the comparison image
            output_path = os.path.join(self.results_dir, "visual_comparison.png")
            cv2.imwrite(output_path, comparison_image)
            
            return output_path
        except Exception as e:
            print(f"Error creating visual comparison: {str(e)}")
            return None
    
    def generate_html_report(self, report, visual_path=None):
        """Generate a formatted HTML report from evaluation data"""
        html_report = f"""
        <h2>Watermark Robustness Evaluation Report</h2>
        <h3>Attack Type: {report['attack_type']}</h3>
        
        <h4>Image Quality Metrics:</h4>
        <ul>
            <li>Contrast: {report['image_quality_metrics']['contrast']:.2f}</li>
            <li>Brightness: {report['image_quality_metrics']['brightness']:.2f}</li>
            <li>Entropy: {report['image_quality_metrics']['entropy']:.2f}</li>
        </ul>
        
        <h4>Watermark Analysis:</h4>
        <ul>
            <li>Average Intensity: {report['watermark_metrics']['intensity']:.2f}</li>
            <li>Contrast: {report['watermark_metrics']['contrast']:.2f}</li>
            <li>Distinct Regions: {report['watermark_metrics']['distinct_regions']}</li>
        </ul>
        
        <h4>Evaluation Summary:</h4>
        """
        
        # Add evaluation summary based on metrics
        watermark_quality = self.evaluate_watermark_quality(report['watermark_metrics'])
        html_report += f"<p><b>{watermark_quality}</b></p>"
        
        # Add recommendation based on quality
        if "Failed" in watermark_quality or "Poor" in watermark_quality:
            html_report += f"""
            <h4>Recommendation:</h4>
            <p>The watermark did not survive the {report['attack_type']} attack well. 
            Consider using a stronger embedding strength or embedding in different frequency bands.</p>
            """
        
        return html_report 