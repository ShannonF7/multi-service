import os
import sys
import torch
import argparse
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Add the project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Load environment variables
load_dotenv()

from src.cv.feature_extractor import get_feature_extractor
from src.database.models_existing import Attraction, AttractionImage
from src.database.session import SessionLocal

def test_verification_logic(image_path, target_attraction_name=None):
    print("1. Connecting to Database...")
    db = SessionLocal()
    
    print("2. Initializing Feature Extractor...")
    extractor = get_feature_extractor()
    
    print(f"3. Loading image from: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    try:
        image = Image.open(image_path).convert('RGB')
        
        print("4. Extracting features...")
        feature_vector = extractor.extract(image)
        print(f"   Feature vector length: {len(feature_vector)}")
        
        print("5. Searching in Database...")
        nearest_image = db.query(AttractionImage).order_by(
            AttractionImage.embedding.l2_distance(feature_vector)
        ).limit(1).first()
        
        if nearest_image:
            attraction = db.query(Attraction).filter(Attraction.id == nearest_image.attraction_id).first()
            identified_name = attraction.name
            print(f"   Match Found!")
            print(f"   Identified Attraction: {identified_name}")
            
            # Verification Logic
            if target_attraction_name:
                is_verified = (target_attraction_name in identified_name) or (identified_name in target_attraction_name)
                print(f"   Target Attraction: {target_attraction_name}")
                print(f"   Verification Result: {'SUCCESS' if is_verified else 'FAILED'}")
            else:
                print("   No target attraction provided for verification.")
            
        else:
            print("   No match found.")
            
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Image Verification Logic")
    
    # Default path logic
    default_image_path = os.path.join(os.path.dirname(current_dir), "figure", "images", "wengcheng.jpg")
    
    parser.add_argument("image_path", nargs="?", default=default_image_path, help="Path to the image file")
    parser.add_argument("--target", "-t", default="瓮城", help="Target attraction name to verify against")
    
    args = parser.parse_args()
    
    test_verification_logic(args.image_path, args.target)
