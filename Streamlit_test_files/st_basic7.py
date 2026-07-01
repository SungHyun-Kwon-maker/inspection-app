import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


st.title("Titanic dataset 분석")

df = sns.load_dataset("titanic")


def show_plot(plot_func, *args, **kwargs):
    fig, ax = plt.subplots(figsize=(7, 4))
    g = plot_func(*args, ax=ax, **kwargs)
    st.pyplot(g.get_figure())
    plt.close(fig)


show_plot(sns.barplot, x="sex", y="survived", data=df)

show_plot(sns.barplot, x="sex", y="fare", data=df)

show_plot(sns.barplot, x="sex", y="survived", hue="class", data=df)

show_plot(
    sns.barplot,
    x="sex",
    y="survived",
    hue="class",
    order=["female", "male"],
    data=df,
)

show_plot(
    sns.barplot,
    x="sex",
    y="survived",
    hue="class",
    order=["female", "male"],
    estimator=sum,
    data=df,
)

show_plot(
    sns.barplot,
    x="sex",
    y="survived",
    hue="class",
    order=["female", "male"],
    palette="Blues_d",
    data=df,
)

fig, ax = plt.subplots(figsize=(7, 4))
g = sns.barplot(x="sex", y="survived", hue="class", data=df, ax=ax)
g.set(xlabel="Gender", ylabel="Survival Rate")
st.pyplot(g.get_figure())
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4))
g = sns.barplot(x="sex", y="survived", hue="class", data=df, ax=ax)
g.set_title("Gender vs Survival Rate in Males and Females")
st.pyplot(g.get_figure())
plt.close(fig)

show_plot(sns.violinplot, x="sex", y="age", hue="survived", data=df, split=True)
