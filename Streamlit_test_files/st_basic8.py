import streamlit as st


if "count" not in st.session_state:
    st.session_state["count"] = 0

st.write(f"카운터 = {st.session_state['count']}")

button = st.button("누르세요")

if button:
    st.session_state["count"] = st.session_state["count"] + 1
    st.rerun()
