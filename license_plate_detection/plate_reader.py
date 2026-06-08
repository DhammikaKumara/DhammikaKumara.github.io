import cv2
import numpy as np
from plate_detector import IndonesianLicensePlateDetector, CharacterRecognizer, ExpiryDetector
import os

class LicensePlateReader:
    """
    Read and recognize characters from detected license plates
    Using pre-trained character recognition model
    Includes expiry detection capabilities
    """
    
    def __init__(self, char_model_path=None):
        """
        Initialize plate reader
        
        Args:
            char_model_path: Path to trained character recognition model
        """
        self.char_recognizer = None
        self.detector = IndonesianLicensePlateDetector()
        self.expiry_detector = ExpiryDetector()
        
        if char_model_path and os.path.exists(char_model_path):
            self.char_recognizer = CharacterRecognizer(char_model_path)
    
    def preprocess_plate(self, plate_region):
        """
        Preprocess plate region for character reading
        
        Args:
            plate_region: Extracted plate image
            
        Returns:
            Preprocessed image
        """
        if len(plate_region.shape) == 3:
            gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_region
        
        # CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Gaussian blur
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        # Threshold
        _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def segment_characters(self, plate_region):
        """
        Segment individual characters from license plate
        
        Args:
            plate_region: Extracted plate image
            
        Returns:
            List of character regions and their positions
        """
        binary = self.preprocess_plate(plate_region)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find contours (characters)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        character_regions = []
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            # Filter based on size (eliminate noise)
            if area > 50 and h > 10 and 3 < w < 30:
                character_regions.append({
                    'region': binary[y:y+h, x:x+w],
                    'original': plate_region[y:y+h, x:x+w],
                    'bbox': (x, y, w, h),
                    'x': x
                })
        
        # Sort by x position (left to right)
        character_regions.sort(key=lambda c: c['x'])
        
        return character_regions
    
    def recognize_character(self, character_image):
        """
        Recognize a single character using trained model
        
        Args:
            character_image: Isolated character image
            
        Returns:
            Recognized character and confidence
        """
        if self.char_recognizer is None:
            raise RuntimeError("Character recognition model not loaded")
        
        character, confidence = self.char_recognizer.recognize_character(character_image)
        return character, confidence
    
    def read_plate(self, plate_region):
        """
        Read the full license plate text
        
        Args:
            plate_region: Extracted plate image
            
        Returns:
            Recognized plate string with confidences
        """
        characters = self.segment_characters(plate_region)
        
        if not characters:
            return "", []
        
        plate_text = ''
        char_data_list = []
        
        for char_data in characters:
            try:
                char, confidence = self.recognize_character(char_data['original'])
                plate_text += char
                char_data_list.append({
                    'char': char,
                    'confidence': confidence,
                    'bbox': char_data['bbox']
                })
            except Exception as e:
                print(f"Error recognizing character: {e}")
                plate_text += '?'
                char_data_list.append({
                    'char': '?',
                    'confidence': 0,
                    'bbox': char_data['bbox']
                })
        
        return plate_text, char_data_list
    
    def check_plate_expiry(self, plate_region, plate_text=None):
        """
        Check if a detected license plate has expired
        
        Args:
            plate_region: Extracted plate image
            plate_text: Recognized plate text (optional, for date extraction)
            
        Returns:
            Expiry detection results
        """
        expiry_result = self.expiry_detector.detect_expiry(
            plate_region,
            plate_text=plate_text,
            use_condition=True,
            use_date=True
        )
        
        return expiry_result
    
    def read_plates_from_image(self, image_path, plate_model_path=None, check_expiry=True):
        """
        Detect and read all plates in an image
        Optionally check for expiry
        
        Args:
            image_path: Path to input image
            plate_model_path: Path to trained plate detection model
            check_expiry: Whether to perform expiry detection
            
        Returns:
            List of detected plates with their text, locations, and expiry status
        """
        if plate_model_path:
            self.detector = IndonesianLicensePlateDetector(plate_model_path)
        
        # Detect plates
        detected_plates, original_image = self.detector.detect_plates(image_path)
        
        results = []
        
        for plate_data in detected_plates:
            plate_region = plate_data['region']
            bbox = plate_data['bbox']
            confidence = plate_data['confidence']
            
            # Read plate text
            plate_text, char_details = self.read_plate(plate_region)
            
            # Check expiry
            expiry_info = None
            if check_expiry:
                expiry_info = self.check_plate_expiry(plate_region, plate_text)
            
            results.append({
                'text': plate_text,
                'bbox': bbox,
                'confidence': confidence,
                'char_details': char_details,
                'region': plate_region,
                'expiry': expiry_info
            })
        
        return results, original_image