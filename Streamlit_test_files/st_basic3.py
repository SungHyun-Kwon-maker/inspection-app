import streamlit as st


st.title("AI 작사가")

if st.button("입력"):
    title = st.text_input("작사할 주제를 제시해주세요")
