import cv2

from preprocess import preprocess_image
from feature_extraction import (
    extract_hog_features,
    show_hog
)

IMAGE_PATH = "../dataset/raw-img/cane/OIP-y_0iDHuMuS8TPnl653QVmgHaJ7.jpeg"

image = cv2.imread(IMAGE_PATH)

processed = preprocess_image(image)

features, hog_image = extract_hog_features(
    processed,
    visualize=True
)

print("="*50)
print("HOG Feature Extraction Successful")
print("="*50)

print("Feature Vector Length :", len(features))

show_hog(processed, hog_image)