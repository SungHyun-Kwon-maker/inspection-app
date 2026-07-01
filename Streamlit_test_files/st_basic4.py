import streamlit as st


st.title("Streamlit text")

code = """
def sample_func():
    print("Sample 함수")
"""
st.code(code, language="python")

st.text("ChatGPT 개발 교육 과정입니다.")
