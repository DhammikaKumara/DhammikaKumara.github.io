import cv2
import os
import sys
from plate_detector import IndonesianLicensePlateDetector, CharacterRecognizer, ExpiryDetector
from plate_reader import LicensePlateReader

def visualize_detection(image, detected_plates):
    """Draw detection results on image"""
    output_image = image.copy()
    
    for idx, plate in enumerate(detected_plates):
        x, y, w, h = plate['bbox']
        confidence = plate.get('confidence', 'N/A')
        
        cv2.rectangle(output_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        if confidence != 'N/A':
            label = f"Plate {idx+1} (Conf: {confidence:.2f})"
        else:
            label = f"Plate {idx+1}"
        
        cv2.putText(output_image, label, (x, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return output_image

def visualize_plate_reading(image, results, show_expiry=True):
    """Draw plate reading results with character confidence and expiry status"""
    output_image = image.copy()
    
    for idx, result in enumerate(results):
        x, y, w, h = result['bbox']
        plate_text = result['text']
        confidence = result['confidence']
        expiry_info = result.get('expiry')
        
        # Draw plate bounding box
        if expiry_info and show_expiry:
            if expiry_info['final_verdict'] == 'expired':
                box_color = (0, 0, 255)
            elif expiry_info['final_verdict'] == 'valid':
                box_color = (0, 255, 0)
            else:
                box_color = (0, 165, 255)
        else:
            box_color = (0, 255, 0)
        
        cv2.rectangle(output_image, (x, y), (x+w, y+h), box_color, 2)
        
        # Draw plate text
        label = f"{plate_text}"
        if confidence:
            label += f" (Conf: {confidence:.2f})"
        
        cv2.putText(output_image, label, (x, y-30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
        
        # Draw expiry status
        if expiry_info and show_expiry:
            expiry_status = expiry_info['final_verdict'].upper()
            expiry_label = f"Expiry: {expiry_status}"
            
            if 'days_until_expiry' in expiry_info:
                days = expiry_info['days_until_expiry']
                expiry_label += f" ({days} days)"
            
            cv2.putText(output_image, expiry_label, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
    
    return output_image

def main():
    """Main function"""
    print("\n" + "="*70)
    print("Indonesian License Plate Detection & Recognition System")
    print("With Expiry Detection Capability")
    print("Using: Canny Edge Detection, HOG Features, and SVM")
    print("="*70 + "\n")
    
    mode = input("Select mode:\n1. Train character recognition model\n2. Train plate detection model\n3. Detect plates\n4. Read plates\n5. Read plates with expiry check\nEnter choice (1-5): ").strip()
    
    if mode == '1':
        print("\n--- Training Character Recognition Model ---\n")
        char_dataset_dir = input("Enter character dataset directory (structure: char_dir/0/, char_dir/1/, etc.): ").strip()
        model_save_path = input("Enter path to save character model (default: ./models/char_recognition.pkl): ").strip() or "./models/char_recognition.pkl"
        
        if not os.path.exists(char_dataset_dir):
            print(f"✗ Character dataset directory not found: {char_dataset_dir}")
            return
        
        try:
            char_recognizer = CharacterRecognizer()
            char_recognizer.train_on_character_dataset(char_dataset_dir, kernel='rbf', C=1.0)
            char_recognizer.save_model(model_save_path)
            print(f"✓ Character model trained and saved to {model_save_path}")
        except Exception as e:
            print(f"✗ Error during training: {e}")
    
    elif mode == '2':
        print("\n--- Training Plate Detection Model ---\n")
        plate_dataset_dir = input("Enter plate dataset directory (directory with bike license plate images): ").strip()
        model_save_path = input("Enter path to save plate model (default: ./models/plate_detection.pkl): ").strip() or "./models/plate_detection.pkl"
        
        if not os.path.exists(plate_dataset_dir):
            print(f"✗ Plate dataset directory not found: {plate_dataset_dir}")
            return
        
        try:
            detector = IndonesianLicensePlateDetector()
            detector.train_svm_on_plate_dataset(plate_dataset_dir, kernel='rbf', C=1.0)
            detector.save_model(model_save_path)
            print(f"✓ Plate detection model trained and saved to {model_save_path}")
        except Exception as e:
            print(f"✗ Error during training: {e}")
    
    elif mode == '3':
        print("\n--- Detect Plates Mode ---\n")
        image_path = input("Enter image path: ").strip()
        plate_model_path = input("Enter plate model path (optional, press Enter to skip): ").strip() or None
        output_path = input("Enter output image path (default: detected_plates.jpg): ").strip() or "detected_plates.jpg"
        
        if not os.path.exists(image_path):
            print(f"✗ Image not found: {image_path}")
            return
        
        try:
            detector = IndonesianLicensePlateDetector(plate_model_path)
            detected_plates, image = detector.detect_plates(image_path)
            
            print(f"\n✓ Detection complete")
            print(f"  Found {len(detected_plates)} potential plate regions")
            
            result_image = visualize_detection(image, detected_plates)
            cv2.imwrite(output_path, result_image)
            print(f"  Result saved to {output_path}\n")
        except Exception as e:
            print(f"✗ Error during detection: {e}")
    
    elif mode == '4':
        print("\n--- Read Plates Mode (Without Expiry Check) ---\n")
        image_path = input("Enter image path: ").strip()
        char_model_path = input("Enter character model path: ").strip()
        plate_model_path = input("Enter plate model path (optional, press Enter to skip): ").strip() or None
        output_path = input("Enter output image path (default: plates_with_text.jpg): ").strip() or "plates_with_text.jpg"
        
        if not os.path.exists(image_path):
            print(f"✗ Image not found: {image_path}")
            return
        
        if not os.path.exists(char_model_path):
            print(f"✗ Character model not found: {char_model_path}")
            return
        
        try:
            reader = LicensePlateReader(char_model_path)
            results, image = reader.read_plates_from_image(image_path, plate_model_path, check_expiry=False)
            
            print(f"\n✓ Reading complete")
            print(f"  Found {len(results)} license plates:")
            
            for idx, result in enumerate(results, 1):
                print(f"\n  Plate {idx}: {result['text']}")
                print(f"    Detection confidence: {result['confidence'] or 'N/A'}")
                print(f"    Character details:")
                for char_info in result['char_details']:
                    print(f"      '{char_info['char']}' (confidence: {char_info['confidence']:.2f})")
            
            result_image = visualize_plate_reading(image, results, show_expiry=False)
            cv2.imwrite(output_path, result_image)
            print(f"\n  Result saved to {output_path}\n")
        except Exception as e:
            print(f"✗ Error during reading: {e}")
    
    elif mode == '5':
        print("\n--- Read Plates Mode (With Expiry Detection) ---\n")
        image_path = input("Enter image path: ").strip()
        char_model_path = input("Enter character model path: ").strip()
        plate_model_path = input("Enter plate model path (optional, press Enter to skip): ").strip() or None
        output_path = input("Enter output image path (default: plates_with_expiry.jpg): ").strip() or "plates_with_expiry.jpg"
        
        if not os.path.exists(image_path):
            print(f"✗ Image not found: {image_path}")
            return
        
        if not os.path.exists(char_model_path):
            print(f"✗ Character model not found: {char_model_path}")
            return
        
        try:
            reader = LicensePlateReader(char_model_path)
            results, image = reader.read_plates_from_image(image_path, plate_model_path, check_expiry=True)
            
            print(f"\n✓ Reading and expiry detection complete")
            print(f"  Found {len(results)} license plates:")
            
            for idx, result in enumerate(results, 1):
                print(f"\n  Plate {idx}: {result['text']}")
                print(f"    Detection confidence: {result['confidence'] or 'N/A'}")
                print(f"    Character details:")
                for char_info in result['char_details']:
                    print(f"      '{char_info['char']}' (confidence: {char_info['confidence']:.2f})")
                
                if result['expiry']:
                    expiry = result['expiry']
                    print(f"\n    Expiry Status:")
                    print(f"      Final Verdict: {expiry['final_verdict'].upper()}")
                    print(f"      Combined Confidence: {expiry['combined_confidence']:.2f}")
                    
                    if expiry['condition_analysis']:
                        print(f"\n      Condition Analysis Method:")
                        print(f"        Status: {expiry['condition_analysis']['status'].upper()}")
                        print(f"        Reason: {expiry['condition_analysis']['reason']}")
                        print(f"        Confidence: {expiry['condition_analysis']['confidence']:.2f}")
                    
                    if expiry['date_analysis']:
                        print(f"\n      Date Analysis Method:")
                        print(f"        Status: {expiry['date_analysis']['status'].upper()}")
                        if 'expiry_date' in expiry['date_analysis']:
                            print(f"        Expiry Date: {expiry['date_analysis']['expiry_date']}")
                            print(f"        Days Until Expiry: {expiry['date_analysis']['days_until_expiry']}")
                        print(f"        Confidence: {expiry['date_analysis']['confidence']:.2f}")
            
            result_image = visualize_plate_reading(image, results, show_expiry=True)
            cv2.imwrite(output_path, result_image)
            print(f"\n  Result saved to {output_path}\n")
        except Exception as e:
            print(f"✗ Error during reading: {e}")
    
    else:
        print("✗ Invalid choice")

if __name__ == "__main__":
    main()