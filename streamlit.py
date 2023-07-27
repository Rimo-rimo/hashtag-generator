import streamlit as st
import requests
from glob import glob
import numpy as np

# ===========================parameter===========================
app_ip = "0.0.0.0"
app_port = "7860"
images = sorted(glob("/home/livin/rimo/genAI/test_images/*"))
# ===========================parameter===========================

# ===========================streamlit ui===========================
st.markdown("<h1 style='text-align: center; color:#5082B3;'>Hash-tag Generator</h1>", unsafe_allow_html=True)

with st.sidebar:
    image_name = images[st.selectbox(":blue[__Select Image__]", list(range(len(images))))]

st.image(image_name)
prompt = st.text_area('**[Prompt]**', 'Please create 7 hashtags for Instagram through a sentence that summarizes the image in one sentence.')
# Please create 7 hashtags for Instagram from this image.

data = {"image":image, "meta_data":None}

if st.button('Generate Hashtag'):
    with st.spinner('Wait for it...'):
        # result = requests.get(f"http://{app_ip}:{app_port}/?image_path={image_name}&prompt={prompt}").json()
        result = requests.post(f"http://{app_ip}:{app_port}/hashtags/", json=data).json()
        st.write(result)