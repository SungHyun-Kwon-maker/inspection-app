import datetime
from datetime import datetime as dt

import streamlit as st


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
        font-size: 1.2rem;
        font-weight: 800;
        margin: 16px 0 8px 0;
    }
    .run-box {
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


def run_box(command):
    st.markdown(f"<div class='run-box'>{command}</div>", unsafe_allow_html=True)


def overview_page():
    hero(
        "Streamlit 사용하기",
        "여러 실습 파일을 하나로 합친 단일 Streamlit 앱입니다.",
    )

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("구성", "1개 파일")
    metric_2.metric("핵심 위젯", "8개 주제")
    metric_3.metric("외부 시각화", "불필요")
    metric_4.metric("배포 파일", "streamlit_app.py")

    section("Streamlit Cloud 설정")
    st.table(
        [
            {"항목": "Repository", "값": "SungHyun-Kwon-maker/inspection-app"},
            {"항목": "Branch", "값": "main"},
            {"항목": "Main file path", "값": "streamlit_app.py"},
        ]
    )

    section("로컬 실행 명령")
    run_box("streamlit run streamlit_app.py")

    section("실습 구성")
    st.dataframe(
        [
            {"파트": "Title", "Streamlit API": "st.title", "내용": "페이지 제목 표시"},
            {"파트": "Input", "Streamlit API": "st.text_input", "내용": "텍스트 입력"},
            {"파트": "Button", "Streamlit API": "st.button", "내용": "버튼 클릭 이벤트"},
            {"파트": "Text", "Streamlit API": "st.code / st.text", "내용": "코드와 텍스트 표시"},
            {"파트": "Data", "Streamlit API": "st.dataframe / st.metric", "내용": "표와 지표 표시"},
            {"파트": "Widgets", "Streamlit API": "checkbox/radio/selectbox", "내용": "입력 위젯"},
            {"파트": "Graph", "Streamlit API": "st.line_chart / st.bar_chart", "내용": "기본 그래프"},
            {"파트": "Session", "Streamlit API": "st.session_state", "내용": "상태 유지"},
        ],
        use_container_width=True,
        hide_index=True,
    )


def title_input_page():
    hero("Title & Input", "제목, 텍스트 입력, 버튼 기능을 한 화면에 정리했습니다.")

    left, right = st.columns([1, 1])
    with left:
        section("Title")
        st.title("This is a title")
        st.title("_Streamlit_ is :blue[cool]")
        st.caption("st.title()은 페이지의 큰 제목을 만들 때 사용합니다.")

    with right:
        section("AI 작사가")
        topic = st.text_input("작사할 주제를 제시해주세요", "코딩")
        mood = st.selectbox("분위기", ["밝게", "잔잔하게", "강렬하게"])
        if st.button("입력", use_container_width=True):
            st.success(f"{topic} 주제로 {mood} 가사를 만들 준비가 됐습니다.")


def text_data_page():
    hero("Text & Data", "코드 블록, 텍스트, 데이터 표, metric을 정리한 화면입니다.")

    code_col, data_col = st.columns([1, 1.35])
    with code_col:
        section("Code Block")
        code = '''def sample_func():
    print("Sample 함수")'''
        st.code(code, language="python")
        st.text("ChatGPT 개발 교육 과정입니다.")

    with data_col:
        section("Sample Data")
        rows = [
            {"week": "1주차", "topic": "title", "score": 72, "status": "done"},
            {"week": "2주차", "topic": "input", "score": 80, "status": "done"},
            {"week": "3주차", "topic": "button", "score": 86, "status": "done"},
            {"week": "4주차", "topic": "chart", "score": 91, "status": "review"},
            {"week": "5주차", "topic": "session", "score": 95, "status": "done"},
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric(label="생산량", value="54000개", delta="-150개")
    metric_2.metric(label="영업이익률", value="18.2%", delta="1.4%")
    metric_3.metric(label="수주잔고", value="3.8억", delta="-0.5억")
    metric_4.metric(label="완료율", value="87%", delta="12%")


def widgets_page():
    hero("UI Widgets", "Streamlit 입력 위젯을 폼처럼 정리했습니다.")

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


def graph_page():
    hero("Graph Widgets", "Streamlit 기본 차트 API로 그래프를 출력합니다.")

    chart_rows = [
        {"week": "1주차", "visitors": 120, "submissions": 18},
        {"week": "2주차", "visitors": 185, "submissions": 31},
        {"week": "3주차", "visitors": 240, "submissions": 45},
        {"week": "4주차", "visitors": 310, "submissions": 66},
        {"week": "5주차", "visitors": 420, "submissions": 82},
        {"week": "6주차", "visitors": 510, "submissions": 103},
    ]

    total_visitors = sum(row["visitors"] for row in chart_rows)
    total_submissions = sum(row["submissions"] for row in chart_rows)

    metric_1, metric_2, metric_3 = st.columns(3)
    metric_1.metric("총 방문", f"{total_visitors:,}")
    metric_2.metric("총 제출", f"{total_submissions:,}")
    metric_3.metric("전환율", f"{total_submissions / total_visitors:.1%}")

    left, right = st.columns(2)
    with left:
        section("방문 추이")
        st.line_chart(chart_rows, x="week", y="visitors", use_container_width=True)
    with right:
        section("제출 수")
        st.bar_chart(chart_rows, x="week", y="submissions", use_container_width=True)


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
    graph_page()
else:
    session_page()
