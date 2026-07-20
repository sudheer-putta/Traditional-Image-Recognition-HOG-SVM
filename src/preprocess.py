import cv2


def preprocess_image(image, image_size=(128, 128)):
    """
    Preprocess a single image.

    Steps:
    1. Resize
    2. Convert to Grayscale
    3. Histogram Equalization
    4. Gaussian Blur
    """

    # Resize
    image = cv2.resize(image, image_size)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Improve contrast
    gray = cv2.equalizeHist(gray)

    # Remove noise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    return gray
