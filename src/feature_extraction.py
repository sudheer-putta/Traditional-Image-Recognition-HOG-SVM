from skimage.feature import hog
import matplotlib.pyplot as plt


def extract_hog_features(
    image,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    visualize=False
):
    """
    Extract HOG features from a grayscale image.

    Parameters
    ----------
    image : ndarray
        Preprocessed grayscale image

    visualize : bool
        If True, returns HOG visualization

    Returns
    -------
    features
    hog_image (optional)
    """

    if visualize:

        features, hog_image = hog(
            image,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            visualize=True,
            block_norm="L2-Hys"
        )

        return features, hog_image

    else:

        features = hog(
            image,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            visualize=False,
            block_norm="L2-Hys"
        )

        return features


def show_hog(original, hog_image):

    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    plt.imshow(original, cmap="gray")
    plt.title("Preprocessed Image")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(hog_image, cmap="gray")
    plt.title("HOG Visualization")
    plt.axis("off")

    plt.tight_layout()
    plt.show()