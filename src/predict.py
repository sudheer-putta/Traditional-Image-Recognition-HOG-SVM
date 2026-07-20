import joblib
from preprocess import preprocess_image
from feature_extraction import extract_hog

model=joblib.load('../models/svm_model.pkl')

def predict(path):
    gray=preprocess_image(path)
    feat,_=extract_hog(gray)
    return model.predict([feat])[0]
