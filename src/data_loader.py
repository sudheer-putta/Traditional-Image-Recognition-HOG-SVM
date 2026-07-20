import os
import cv2
import numpy as np
from collections import Counter

# Italian folder names -> English labels
LABEL_MAP = {
    "cane": "dog",
    "cavallo": "horse",
    "elefante": "elephant",
    "farfalla": "butterfly",
    "gallina": "chicken",
    "gatto": "cat",
    "mucca": "cow",
    "pecora": "sheep",
    "ragno": "spider",
    "scoiattolo": "squirrel"
}


def load_dataset(dataset_path, image_size=(128, 128)):
    """
    Loads all images from dataset folder.

    Returns:
        images : numpy array
        labels : numpy array
    """

    images = []
    labels = []

    print("=" * 60)
    print("Loading Dataset...")
    print("=" * 60)

    for folder in sorted(os.listdir(dataset_path)):

        folder_path = os.path.join(dataset_path, folder)

        if not os.path.isdir(folder_path):
            continue

        english_label = LABEL_MAP.get(folder, folder)

        image_count = 0

        for file in os.listdir(folder_path):

            file_path = os.path.join(folder_path, file)

            image = cv2.imread(file_path)

            if image is None:
                continue

            image = cv2.resize(image, image_size)

            images.append(image)

            labels.append(english_label)

            image_count += 1

        print(f"{english_label:<12} : {image_count} images")

    print("\nDataset Loaded Successfully.")

    return np.array(images), np.array(labels)


if __name__ == "__main__":

    DATASET_PATH = "../dataset/raw-img"

    X, y = load_dataset(DATASET_PATH)

    print("\nDataset Shape")
    print("----------------------------")
    print("Images :", X.shape)
    print("Labels :", y.shape)

    print("\nUnique Classes")
    print("----------------------------")
    print(np.unique(y))

    print("\nSamples per Class")
    print("----------------------------")

    counter = Counter(y)

    for label, count in counter.items():
        print(f"{label:<12} : {count}")