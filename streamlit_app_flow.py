import streamlit as st
import docx2txt
import os
import shutil
import utils

st.set_page_config(
    page_title="Vietnamese Punctuation Restoration", page_icon="📊", initial_sidebar_state="expanded"
)

st.header(
    """
Vietnamese Punctuation Restoration
"""
)


@st.cache
def reset_folder():
    print("reset_folder")
    dir_path = os.path.dirname(os.path.realpath(__file__))
    tmpPath = os.path.join(dir_path, 'tmp')
    if os.path.exists(tmpPath):
        shutil.rmtree(tmpPath)
    if not os.path.exists(tmpPath):
        os.mkdir(tmpPath)


reset_folder()

text_input = st.text_area('Text to translate')

uploaded_file = st.file_uploader("Upload document", type=['docx'])

if st.button('Predict'):
    print("text_input")
    print(text_input)
    print("uploaded_file")
    print(uploaded_file)

    if uploaded_file is None and text_input == "":
        print("none")
        st.write("text_input or uploaded_file cant be none")
        st.stop()

    result = dict()
    if text_input.strip() != "":
        print("model predict on text_input")
        txt = utils.preprocessing(text_input)
        txt = utils.inference(txt)
        txt = txt.replace("_", " ")
        txt = txt[:-1]
        result["Result of Text to translate"] = txt
        # result["text_input"] = text_input
    if uploaded_file is not None:
        file_location = f"tmp/{uploaded_file.name}"
        with open(file_location, "wb+") as file_object:
            file_object.write(uploaded_file.getbuffer())
        print(f"info: file {uploaded_file.name} saved at {file_location}")
        text_uploaded_file = docx2txt.process(file_location)
        print("text_uploaded_file")
        print(text_uploaded_file)
        print("model predict on uploaded_file")
        txt = utils.preprocessing(text_uploaded_file)
        txt = utils.inference(txt)
        txt = txt.replace("_", " ")
        txt = txt[:-1]
        result["Result of Upload document"] = txt
        # result["uploaded_file"] = text_uploaded_file

    for key, value in result.items():
        st.write(key)
        st.write(value)
