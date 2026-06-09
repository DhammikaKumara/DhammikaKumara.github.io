import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from skimage.feature import hog
import joblib
import os
from datetime import datetime

class CharacterRecognizer:
    """
    Character Recognition System using HOG + SVM
    Trained on character dataset (A-Z, 0-9)
    """
    
    def __init__(self, model_path=None):
        """
        Initialize character recognizer
        
        Args:
            model_path: Path to pre-trained character SVM model
        """
        self.svm_model = None
        self.scaler = None
        self.label_encoder = None
        self.hog_params = {
            'orientations': 9,
            'pixels_per_cell': (8, 8),
            'cells_per_block': (2, 2),
            'block_norm': 'L2-Hys'
        }
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def extract_hog_features(self, image):
        """
        Extract HOG features from character image
        
        Args:
            image: Character image
            
        Returns:
            HOG feature vector
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize to standard size
        image = cv2.resize(image, (32, 32))
        
        # Normalize
        image = image / 255.0
        
        # Extract HOG
        features = hog(image, **self.hog_params)
        
        return features
    
    def train_on_character_dataset(self, dataset_dir, kernel='rbf', C=1.0):
        """
        Train multi-class SVM on character dataset
        
        Expected directory structure:
        dataset_dir/
        ├── 0/
        ├── 1/
        ├── A/
        ├── B/
        └── ...
        
        Args:
            dataset_dir: Root directory containing character folders
            kernel: SVM kernel type
            C: Regularization parameter
        """
        print("\n=== Training Character Recognition System ===")
        print(f"Loading character dataset from {dataset_dir}\n")
        
        features_list = []
        labels_list = []
        character_names = []
        
        # Iterate through character directories
        char_dirs = sorted([d for d in os.listdir(dataset_dir) 
                           if os.path.isdir(os.path.join(dataset_dir, d))])
        
        print(f"Found {len(char_dirs)} character classes: {', '.join(char_dirs)}\n")
        total_samples = 0
        
        for char_label, char_dir in enumerate(char_dirs):
            char_path = os.path.join(dataset_dir, char_dir)
            sample_count = 0
            
            # Load images from character directory
            for filename in os.listdir(char_path):
                if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    filepath = os.path.join(char_path, filename)
                    image = cv2.imread(filepath)
                    
                    if image is not None:
                        # Extract features from original
                        features = self.extract_hog_features(image)
                        features_list.append(features)
                        labels_list.append(char_label)
                        sample_count += 1
                        
                        # Data augmentation: slight rotations and scale
                        for angle in [-10, 10]:
                            h, w = image.shape[:2]
                            center = (w // 2, h // 2)
                            M = cv2.getRotationMatrix2D(center, angle, 1.0)
                            rotated = cv2.warpAffine(image, M, (w, h))
                            features = self.extract_hog_features(rotated)
                            features_list.append(features)
                            labels_list.append(char_label)
                            sample_count += 1
            
            character_names.append(char_dir)
            print(f"  Character '{char_dir}': {sample_count} samples (with augmentation)")
            total_samples += sample_count
        
        if not features_list:
            raise ValueError("No training samples found in character dataset")
        
        print(f"\nTotal training samples: {total_samples}\n")
        
        # Convert to numpy arrays
        X = np.array(features_list)
        y = np.array(labels_list)
        
        # Standardize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Setup label encoder
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(character_names)
        
        # Train multi-class SVM
        print(f"Training multi-class SVM (kernel={kernel}, C={C})...")
        self.svm_model = SVC(kernel=kernel, C=C, probability=True)
        self.svm_model.fit(X_scaled, y)
        
        print("✓ Character recognition model trained successfully")
        print(f"  Classes: {character_names}\n")
    
    def recognize_character(self, character_image):
        """
        Recognize a single character
        
        Args:
            character_image: Cropped character image
            
        Returns:
            Recognized character and confidence
        """
        if self.svm_model is None:
            raise RuntimeError("Character model not trained or loaded")
        
        features = self.extract_hog_features(character_image)
        features = features.reshape(1, -1)
        
        if self.scaler:
            features = self.scaler.transform(features)
        
        prediction = self.svm_model.predict(features)[0]
        probabilities = self.svm_model.predict_proba(features)[0]
        confidence = np.max(probabilities)
        
        character = self.label_encoder.classes_[prediction]
        
        return character, confidence
    
    def save_model(self, model_path):
        """Save trained character recognition model"""
        if self.svm_model is None:
            raise RuntimeError("No trained model to save")
        
        os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)
        
        joblib.dump({
            'model': self.svm_model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder
        }, model_path)
        
        print(f"Character model saved to {model_path}\n")
    
    def load_model(self, model_path):
        """Load pre-trained character recognition model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        data = joblib.load(model_path)
        self.svm_model = data['model']
        self.scaler = data['scaler']
        self.label_encoder = data['label_encoder']
        
        print(f"Character model loaded from {model_path}\n")


class ExpiryDetector:
    """
    Detect if an Indonesian motorcycle license plate has expired
    Based on plate condition analysis and date interpretation
    
    Indonesian motorcycle plates are black characters on white background.
    Expired plates show signs of deterioration, fading, or darkening.
    """
    
    def __init__(self):
        """Initialize expiry detector"""
        self.expiry_month_detector = None
        self.expiry_year_detector = None
    
    def analyze_plate_condition(self, plate_region):
        """
        Analyze plate condition to detect expiry status
        
        Expired plates typically show:
        - Overall darkening/fading
        - Uneven color distribution
        - Character blur or deterioration
        - Higher contrast degradation
        
        Args:
            plate_region: Extracted plate image
            
        Returns:
            Plate condition analysis results
        """
        if len(plate_region.shape) == 3:
            gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_region
        
        # Analyze overall brightness
        mean_brightness = np.mean(gray)
        
        # Analyze contrast (standard deviation)
        contrast = np.std(gray)
        
        # Check for white background presence
        # Good plates should have significant white area (high pixel values)
        white_pixels = np.sum(gray > 200)
        total_pixels = gray.shape[0] * gray.shape[1]
        white_ratio = white_pixels / total_pixels
        
        # Check for black character presence
        # Good plates should have significant black area (low pixel values)
        black_pixels = np.sum(gray < 100)
        black_ratio = black_pixels / total_pixels
        
        # Analyze edge sharpness (valid plates have sharp edges)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / total_pixels
        
        return {
            'mean_brightness': mean_brightness,
            'contrast': contrast,
            'white_ratio': white_ratio,
            'black_ratio': black_ratio,
            'edge_density': edge_density
        }
    
    def extract_expiry_date_from_plate(self, plate_text):
        """
        Extract expiry date information from recognized plate text
        
        Indonesian motorcycle plates: [NUMBER] [LETTERS]
        May contain expiry month/year information
        
        Args:
            plate_text: Recognized plate text
            
        Returns:
            Extracted date information
        """
        import re
        
        # Look for date patterns (MM/YY or MM-YY)
        date_pattern = r'(\d{1,2})[/-](\d{1,2})'
        matches = re.findall(date_pattern, plate_text)
        
        dates = []
        for match in matches:
            month = int(match[0])
            year = int(match[1])
            
            # Validate month (1-12)
            if 1 <= month <= 12:
                dates.append({
                    'month': month,
                    'year': year,
                    'text': f"{month:02d}/{year:02d}"
                })
        
        return dates
    
    def check_expiry_by_condition(self, plate_region):
        """
        Check if plate has expired based on physical condition
        
        Valid plates: Good brightness, high contrast, sharp edges
        Expired plates: Darkened, low contrast, blurred edges
        
        Args:
            plate_region: Extracted plate image
            
        Returns:
            Expiry status and confidence
        """
        condition = self.analyze_plate_condition(plate_region)
        
        # Scoring system for plate condition
        brightness_score = 1.0 if condition['mean_brightness'] > 180 else 0.5 if condition['mean_brightness'] > 150 else 0.0
        contrast_score = 1.0 if condition['contrast'] > 40 else 0.5 if condition['contrast'] > 25 else 0.0
        white_score = 1.0 if condition['white_ratio'] > 0.4 else 0.5 if condition['white_ratio'] > 0.2 else 0.0
        edge_score = 1.0 if condition['edge_density'] > 0.08 else 0.5 if condition['edge_density'] > 0.05 else 0.0
        
        # Combined score
        condition_score = (brightness_score + contrast_score + white_score + edge_score) / 4.0
        
        if condition_score < 0.4:
            return {
                'status': 'expired',
                'method': 'condition_analysis',
                'confidence': abs(1.0 - condition_score),
                'reason': 'Plate shows signs of deterioration (dark, low contrast, faded)'
            }
        else:
            return {
                'status': 'valid',
                'method': 'condition_analysis',
                'confidence': condition_score,
                'reason': 'Plate appears well-maintained with good contrast and clarity'
            }
    
    def check_expiry_by_date(self, plate_text):
        """
        Check if plate has expired based on extracted date
        
        Args:
            plate_text: Recognized plate text
            
        Returns:
            Expiry status if date found
        """
        dates = self.extract_expiry_date_from_plate(plate_text)
        
        if not dates:
            return {
                'status': 'unknown',
                'method': 'date_extraction',
                'confidence': 0,
                'reason': 'No date found in plate text'
            }
        
        current_date = datetime.now()
        results = []
        
        for date_info in dates:
            month = date_info['month']
            year = date_info['year']
            
            # Interpret 2-digit year (assume 20xx for recent years)
            if year < 100:
                full_year = 2000 + year if year < 50 else 1900 + year
            else:
                full_year = year
            
            # Create expiry date (last day of the month)
            if month == 12:
                expiry_date = datetime(full_year + 1, 1, 1)
            else:
                expiry_date = datetime(full_year, month + 1, 1)
            
            is_expired = current_date > expiry_date
            days_until_expiry = (expiry_date - current_date).days
            
            results.append({
                'status': 'expired' if is_expired else 'valid',
                'method': 'date_extraction',
                'date': date_info['text'],
                'expiry_date': expiry_date.strftime('%Y-%m-%d'),
                'days_until_expiry': days_until_expiry,
                'confidence': 0.8
            })
        
        return results[0] if results else None
    
    def detect_expiry(self, plate_region, plate_text=None, use_condition=True, use_date=True):
        """
        Comprehensive expiry detection using multiple methods
        
        Args:
            plate_region: Extracted plate image
            plate_text: Recognized plate text (optional)
            use_condition: Use condition analysis method
            use_date: Use date extraction method
            
        Returns:
            Expiry detection result with combined confidence
        """
        results = {
            'condition_analysis': None,
            'date_analysis': None,
            'final_verdict': None,
            'combined_confidence': 0
        }
        
        # Method 1: Condition analysis (plate deterioration)
        if use_condition:
            results['condition_analysis'] = self.check_expiry_by_condition(plate_region)
        
        # Method 2: Date extraction
        if use_date and plate_text:
            results['date_analysis'] = self.check_expiry_by_date(plate_text)
        
        # Combine results
        confidences = []
        expired_votes = 0
        valid_votes = 0
        
        if results['condition_analysis']:
            confidences.append(results['condition_analysis']['confidence'])
            if results['condition_analysis']['status'] == 'expired':
                expired_votes += 1
            else:
                valid_votes += 1
        
        if results['date_analysis']:
            confidences.append(results['date_analysis']['confidence'])
            if results['date_analysis']['status'] == 'expired':
                expired_votes += 1
            else:
                valid_votes += 1
        
        # Final verdict based on voting
        if expired_votes > valid_votes:
            results['final_verdict'] = 'expired'
        elif valid_votes > expired_votes:
            results['final_verdict'] = 'valid'
        else:
            results['final_verdict'] = 'unknown'
        
        results['combined_confidence'] = np.mean(confidences) if confidences else 0
        
        return results


class IndonesianLicensePlateDetector:
    """
    Indonesian License Plate Detection System
    Using Canny Edge Detection, HOG Features, and SVM Classifier
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the detector
        
        Args:
            model_path: Path to pre-trained plate detection SVM model
        """
        self.svm_model = None
        self.scaler = None
        self.hog_params = {
            'orientations': 9,
            'pixels_per_cell': (8, 8),
            'cells_per_block': (2, 2),
            'block_norm': 'L2-Hys'
        }
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def preprocess_image(self, image_path):
        """
        Load and preprocess image for plate detection
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Original image and grayscale version
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image from {image_path}")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image, gray
    
    def apply_canny_edge_detection(self, gray_image, threshold1=100, threshold2=200):
        """
        Apply Canny edge detection
        
        Args:
            gray_image: Grayscale image
            threshold1: Lower threshold
            threshold2: Upper threshold
            
        Returns:
            Edge-detected image
        """
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
        edges = cv2.Canny(blurred, threshold1, threshold2)
        return edges
    
    def find_plate_contours(self, edges):
        """Find contours in edge-detected image"""
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        return contours
    
    def extract_plate_regions(self, image, contours, min_area=500, max_area=None):
        """
        Extract rectangular regions that could be license plates
        
        Args:
            image: Original image
            contours: List of contours
            min_area: Minimum area threshold
            max_area: Maximum area threshold
            
        Returns:
            List of potential plate regions and bounding boxes
        """
        plate_regions = []
        bounding_boxes = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < min_area:
                continue
            
            if max_area and area > max_area:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by aspect ratio (Indonesian plates: 3.5:1 to 5:1)
            aspect_ratio = w / h if h != 0 else 0
            if not (3.0 <= aspect_ratio <= 5.5):
                continue
            
            region = image[y:y+h, x:x+w]
            plate_regions.append(region)
            bounding_boxes.append((x, y, w, h))
        
        return plate_regions, bounding_boxes
    
    def extract_hog_features(self, image):
        """Extract HOG features from image"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        image = cv2.resize(image, (128, 32))
        features = hog(image, **self.hog_params)
        
        return features
    
    def classify_plate_region(self, region):
        """Classify if a region is a license plate"""
        if self.svm_model is None:
            raise RuntimeError("SVM model not trained or loaded")
        
        features = self.extract_hog_features(region)
        features = features.reshape(1, -1)
        
        if self.scaler:
            features = self.scaler.transform(features)
        
        prediction = self.svm_model.predict(features)[0]
        confidence = abs(self.svm_model.decision_function(features)[0])
        
        return prediction, confidence
    
    def detect_plates(self, image_path, confidence_threshold=0.5):
        """Main detection pipeline"""
        image, gray = self.preprocess_image(image_path)
        edges = self.apply_canny_edge_detection(gray)
        contours = self.find_plate_contours(edges)
        plate_regions, bounding_boxes = self.extract_plate_regions(image, contours)
        
        detected_plates = []
        
        for idx, (region, bbox) in enumerate(zip(plate_regions, bounding_boxes)):
            if self.svm_model:
                prediction, confidence = self.classify_plate_region(region)
                
                if prediction == 1 and confidence >= confidence_threshold:
                    detected_plates.append({
                        'region': region,
                        'bbox': bbox,
                        'confidence': confidence,
                        'index': idx
                    })
            else:
                detected_plates.append({
                    'region': region,
                    'bbox': bbox,
                    'confidence': None,
                    'index': idx
                })
        
        return detected_plates, image
    
    def train_svm_on_plate_dataset(self, plate_dataset_dir, kernel='rbf', C=1.0):
        """
        Train SVM using plate images as positive samples
        Generate synthetic negatives from random crops
        
        Args:
            plate_dataset_dir: Directory with license plate images
            kernel: SVM kernel type
            C: Regularization parameter
        """
        print("\n=== Training Plate Detection System ===")
        print(f"Loading plate dataset from {plate_dataset_dir}\n")
        
        features_list = []
        labels_list = []
        
        # Load positive samples
        print("Loading positive samples (actual license plates)...")
        positive_count = 0
        
        if os.path.exists(plate_dataset_dir):
            for filename in os.listdir(plate_dataset_dir):
                if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    filepath = os.path.join(plate_dataset_dir, filename)
                    image = cv2.imread(filepath)
                    if image is not None:
                        # Original
                        features = self.extract_hog_features(image)
                        features_list.append(features)
                        labels_list.append(1)
                        positive_count += 1
                        
                        # Augmentation: rotations
                        for angle in [-15, -10, 10, 15]:
                            h, w = image.shape[:2]
                            center = (w // 2, h // 2)
                            M = cv2.getRotationMatrix2D(center, angle, 1.0)
                            rotated = cv2.warpAffine(image, M, (w, h))
                            features = self.extract_hog_features(rotated)
                            features_list.append(features)
                            labels_list.append(1)
                            positive_count += 1
        
        print(f"  Loaded {positive_count} positive samples (with augmentation)\n")
        
        # Generate synthetic negatives
        print("Generating synthetic negative samples...")
        negative_count = 0
        
        if os.path.exists(plate_dataset_dir):
            for filename in os.listdir(plate_dataset_dir):
                if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    filepath = os.path.join(plate_dataset_dir, filename)
                    image = cv2.imread(filepath)
                    if image is not None:
                        h, w = image.shape[:2]
                        # Random crops as negatives
                        for _ in range(3):
                            y_start = np.random.randint(0, max(1, h - 32))
                            x_start = np.random.randint(0, max(1, w - 128))
                            crop = image[y_start:y_start+32, x_start:x_start+128]
                            
                            if crop.shape[0] >= 32 and crop.shape[1] >= 128:
                                features = self.extract_hog_features(crop)
                                features_list.append(features)
                                labels_list.append(0)
                                negative_count += 1
        
        print(f"  Generated {negative_count} synthetic negative samples\n")
        
        if not features_list:
            raise ValueError("No training samples found")
        
        X = np.array(features_list)
        y = np.array(labels_list)
        
        print(f"Training SVM...")
        print(f"  Total samples: {len(features_list)}")
        print(f"  Positive: {np.sum(y)}")
        print(f"  Negative: {len(y) - np.sum(y)}\n")
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.svm_model = SVC(kernel=kernel, C=C, probability=True)
        self.svm_model.fit(X_scaled, y)
        
        print(f"✓ Plate detection model trained successfully\n")
    
    def save_model(self, model_path):
        """Save trained model"""
        if self.svm_model is None:
            raise RuntimeError("No trained model to save")
        
        os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)
        joblib.dump({
            'model': self.svm_model,
            'scaler': self.scaler
        }, model_path)
        
        print(f"Plate detection model saved to {model_path}\n")
    
    def load_model(self, model_path):
        """Load pre-trained model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        data = joblib.load(model_path)
        self.svm_model = data['model']
        self.scaler = data['scaler']
        
        print(f"Plate detection model loaded from {model_path}\n")