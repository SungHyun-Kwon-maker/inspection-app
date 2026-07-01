import streamlit as st


st.title("AI 작사가")

title = st.text_input("작사할 주제를 제시해주세요", "코딩")
st.write("작사할 주제는", title)
