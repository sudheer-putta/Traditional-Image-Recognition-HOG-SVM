import cv2

def preprocess_image(path, size=(128,128)):
    img=cv2.imread(path)
    img=cv2.resize(img,size)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return gray
