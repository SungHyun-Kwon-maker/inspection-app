import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris


iris = load_iris()

dataframe = pd.DataFrame(iris.data, columns=iris.feature_names)
dataframe["target"] = iris.target

st.dataframe(dataframe, use_container_width=True)
st.table(dataframe)

st.metric(label="생산량", value="54000개", delta="-150개")
st.metric(label="영업이익률", value="18.2%", delta="1.4%")

sales_1, sales_2 = st.columns(2)
sales_1.metric(label="수주잔고", value="3.8억", delta="-0.5억")
sales_2.metric(label="수주잔고", value="2.5억", delta="5000천만")
