import streamlit as st
st.title('Traditional Image Recognition (HOG + SVM)')
st.write('Train the model first, then load it in the app.')
uploaded=st.file_uploader('Upload image')
if uploaded:
    st.image(uploaded)
