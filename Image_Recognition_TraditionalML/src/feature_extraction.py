from skimage.feature import hog

def extract_hog(gray):
    feat,vis=hog(gray,orientations=9,pixels_per_cell=(8,8),
                 cells_per_block=(2,2),visualize=True)
    return feat,vis
