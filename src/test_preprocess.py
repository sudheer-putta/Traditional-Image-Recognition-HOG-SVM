import cv2
import matplotlib.pyplot as plt
from preprocess import preprocess_image

IMAGE_PATH = "../dataset/raw-img/cane/OIP-y_0iDHuMuS8TPnl653QVmgHaJ7.jpeg"

image = cv2.imread(IMAGE_PATH)

processed = preprocess_image(image)

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(processed, cmap="gray")
plt.title("Processed")
plt.axis("off")

plt.show()