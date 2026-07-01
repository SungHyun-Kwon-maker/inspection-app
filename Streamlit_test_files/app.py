import datetime
from datetime import datetime as dt

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.datasets import load_iris


st.set_page_config(
    page_title="Streamlit Practice Lab",
    page_icon="S",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background: #f5f7fb;
    }
    [data-testid="stSidebar"] {
        background: #111827;
    }
    [data-testid="stSidebar"] * {
        color: #f9fafb;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    .hero {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-left: 8px solid #ef4444;
        border-radius: 8px;
        padding: 24px 28px;
        margin-bottom: 18px;
        box-shadow: 0 8px 24px rgba(16, 24, 40, 0.06);
    }
    .hero h1 {
        color: #111827;
        font-size: 2.35rem;
        line-height: 1.15;
        margin: 0 0 10px 0;
        letter-spacing: 0;
    }
    .hero p {
        color: #4b5563;
        font-size: 1rem;
        margin: 0;
    }
    .section-title {
        color: #111827;
        font-size: 1.25rem;
        font-weight: 800;
        margin: 16px 0 8px 0;
    }
    .code-note {
        background: #111827;
        color: #f9fafb;
        border-radius: 8px;
        padding: 14px 16px;
        font-family: monospace;
        font-size: 0.9rem;
        margin: 8px 0 16px 0;
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 14px 16px;
        box-shadow: 0 6px 16px rgba(16, 24, 40, 0.05);
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_iris_frame():
    iris = load_iris()
    dataframe = pd.DataFrame(iris.data, columns=iris.feature_names)
    dataframe["target"] = iris.target
    dataframe["species"] = dataframe["target"].map(
        {idx: name for idx, name in enumerate(iris.target_names)}
    )
    return dataframe


@st.cache_data
def make_chart_data():
    return pd.DataFrame(
        {
            "week": ["1주차", "2주차", "3주차", "4주차", "5주차", "6주차"],
            "visitors": [120, 185, 240, 310, 420, 510],
            "submissions": [18, 31, 45, 66, 82, 103],
            "category": ["title", "input", "button", "data", "chart", "session"],
        }
    )


def hero(title, subtitle):
    st.markdown(
        f"""
        <div class="hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title):
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)


def command(path):
    st.markdown(
        f"<div class='code-note'>streamlit run {path}</div>",
        unsafe_allow_html=True,
    )


def overview_page():
    hero(
        "Streamlit 사용하기",
        "st_basic1.py부터 st_basic8.py까지의 실습을 한 화면에서 탐색하는 통합 UI입니다.",
    )

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("실습 파일", "8개")
    metric_2.metric("핵심 위젯", "Title/Input/Button")
    metric_3.metric("데이터 표시", "DataFrame/Metric")
    metric_4.metric("상태 관리", "Session State")

    section("실행 명령")
    command("Streamlit_test_files/app.py")

    section("실습 구성")
    roadmap = pd.DataFrame(
        [
            ["st_basic1.py", "st.title", "페이지 제목 출력"],
            ["st_basic2.py", "st.text_input", "텍스트 입력값 표시"],
            ["st_basic3.py", "st.button", "버튼 클릭 이벤트"],
            ["st_basic4.py", "st.code / st.text", "코드와 텍스트 표시"],
            ["st_basic5.py", "st.dataframe / st.metric", "데이터와 지표 출력"],
            ["st_basic6.py", "UI widgets", "체크박스, 라디오, 슬라이더"],
            ["st_basic7.py", "st.pyplot", "그래프 출력"],
            ["st_basic8.py", "st.session_state", "카운터 상태 유지"],
        ],
        columns=["파일", "핵심 API", "내용"],
    )
    st.dataframe(roadmap, use_container_width=True, hide_index=True)


def title_input_page():
    hero("Title & Input", "제목, 텍스트 입력, 버튼을 보기 좋게 배치한 화면입니다.")

    left, right = st.columns([1, 1])
    with left:
        section("Title")
        st.title("This is a title")
        st.title("_Streamlit_ is :blue[cool]")
        command("Streamlit_test_files/st_basic1.py")

    with right:
        section("Input")
        topic = st.text_input("작사할 주제를 제시해주세요", "코딩")
        if st.button("입력", use_container_width=True):
            st.success(f"작사할 주제는 {topic}입니다.")
        command("Streamlit_test_files/st_basic2.py")


def text_data_page():
    hero("Text & Data", "코드 블록, 표, 지표를 대시보드 스타일로 정리했습니다.")

    code_col, data_col = st.columns([1, 1.3])
    with code_col:
        section("Code Block")
        code = '''def sample_func():
    print("Sample 함수")'''
        st.code(code, language="python")
        st.text("ChatGPT 개발 교육 과정입니다.")
        command("Streamlit_test_files/st_basic4.py")

    with data_col:
        section("Iris Data")
        iris_df = load_iris_frame()
        st.dataframe(iris_df, use_container_width=True, height=260)
        command("Streamlit_test_files/st_basic5.py")

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric(label="생산량", value="54000개", delta="-150개")
    metric_2.metric(label="영업이익률", value="18.2%", delta="1.4%")
    metric_3.metric(label="수주잔고", value="3.8억", delta="-0.5억")
    metric_4.metric(label="수주잔고", value="2.5억", delta="5000천만")


def widgets_page():
    hero("UI Widgets", "Streamlit 입력 위젯을 폼처럼 정리한 화면입니다.")

    left, right = st.columns([1, 1])
    with left:
        if st.button("버튼", use_container_width=True):
            st.info("버튼이 눌렸습니다")

        human = st.checkbox("사람이면 체크해주세요.")
        if human:
            st.success("당신은 사람이군요!")

        religion = st.radio(
            index=None,
            label="당신의 종교는 무엇입니까?",
            options=("기독교", "천주교", "불교", "기타", "무교"),
        )
        if religion:
            st.write("당신의 종교는 *" + religion + "* 이군요!")

        school = st.selectbox(
            index=None,
            label="당신의 최종학력은 무엇입니까?",
            options=("대학원졸", "대졸", "고졸"),
        )
        if school:
            st.write("당신의 최종학력은 :sparkle:" + school + ":sparkle: 이군요!")

    with right:
        foods = st.multiselect(
            "당신이 가장 좋아하는 음식은 뭔가요?",
            ["돼지갈비", "소갈비", "스테이크", "생선회", "삼겹살", "김치찌개"],
        )
        if foods:
            st.write(f"당신이 가장 좋아하는 음식은 {foods}입니다.")

        bp = st.slider("혈압 범위를 지정해주세요.", 6.0, 200.0, (90.0, 130.0))
        st.write("이완기 혈압:", bp[0])
        st.write("수축기 혈압:", bp[1])

        birthday_time = st.slider(
            "당신의 출생년월일 시각을 알려주세요",
            min_value=dt(1950, 1, 1, 0, 0),
            max_value=dt(2024, 3, 11, 12, 0),
            step=datetime.timedelta(hours=1),
            format="MM/DD/YY - HH:mm",
        )
        st.write(f"당신의 생년월일시는 {birthday_time}입니다.")

    command("Streamlit_test_files/st_basic6.py")


def chart_page():
    hero("Graph Widgets", "샘플 데이터를 seaborn 그래프로 만들고 st.pyplot으로 출력합니다.")

    chart_df = make_chart_data()
    metric_1, metric_2, metric_3 = st.columns(3)
    metric_1.metric("총 방문", f"{chart_df['visitors'].sum():,}")
    metric_2.metric("총 제출", f"{chart_df['submissions'].sum():,}")
    metric_3.metric("전환율", f"{chart_df['submissions'].sum() / chart_df['visitors'].sum():.1%}")

    left, right = st.columns(2)
    with left:
        section("주차별 방문 추이")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.lineplot(data=chart_df, x="week", y="visitors", marker="o", ax=ax)
        ax.set_xlabel("Week")
        ax.set_ylabel("Visitors")
        ax.grid(axis="y", alpha=0.25)
        sns.despine(fig=fig)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with right:
        section("실습별 제출 수")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(data=chart_df, x="category", y="submissions", palette="Set2", ax=ax)
        ax.set_xlabel("Practice")
        ax.set_ylabel("Submissions")
        ax.grid(axis="y", alpha=0.25)
        sns.despine(fig=fig)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    command("Streamlit_test_files/st_basic7.py")


def session_page():
    hero("Session State", "버튼을 눌러도 값이 유지되는 카운터 예제입니다.")

    if "count" not in st.session_state:
        st.session_state["count"] = 0

    count_col, button_col = st.columns([2, 1])
    count_col.metric("카운터", st.session_state["count"])

    if button_col.button("누르세요", use_container_width=True):
        st.session_state["count"] += 1
        st.rerun()

    if st.button("초기화", use_container_width=True):
        st.session_state["count"] = 0
        st.rerun()

    command("Streamlit_test_files/st_basic8.py")


st.sidebar.title("Streamlit Lab")
page = st.sidebar.radio(
    "페이지",
    [
        "Overview",
        "Title & Input",
        "Text & Data",
        "Widgets",
        "Graph",
        "Session",
    ],
)

if page == "Overview":
    overview_page()
elif page == "Title & Input":
    title_input_page()
elif page == "Text & Data":
    text_data_page()
elif page == "Widgets":
    widgets_page()
elif page == "Graph":
    chart_page()
else:
    session_page()
