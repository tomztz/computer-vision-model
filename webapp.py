import streamlit as st
import numpy as np
import pandas as pd
import detect_video as dv
import sys

st.title('Countify+')


uploaded_file = st.file_uploader("Choose a file")
vid = dv.main(video=uploaded_file)
#print(str(sys.argv))
video_file = open(vid.output, "rb")
video_bytes = video_file.read()
st.video(video_bytes)
# name = st.text_input('Name')
# if not name:
#    st.warning('Please input a name.')
#    st.stop()
# st.success('Thank you for inputting a name.')
