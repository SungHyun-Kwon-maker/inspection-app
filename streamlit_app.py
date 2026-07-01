from io import BytesIO
from pathlib import Path

import numpy as np
import openvino as ov
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError


APP_DIR = Path(__file__).resolve().parent
MODEL_XML = APP_DIR / "weights" / "leather_model.xml"
MODEL_BIN = MODEL_XML.with_suffix(".bin")
INPUT_SIZE = (224, 224)
CLASS_NAMES = ("정상", "불량")


st.set_page_config(
    page_title="Leather Inspection",
    page_icon="I",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background: #f4f6f8;
    }
    [data-testid="stSidebar"] {
        background: #121826;
    }
    [data-testid="stSidebar"] * {
        color: #f9fafb;
    }
    .block-container {
        padding-top: 1.8rem;
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
        line-height: 1.12;
        margin: 0 0 10px 0;
        letter-spacing: 0;
    }
    .hero p {
        color: #4b5563;
        font-size: 1rem;
        margin: 0;
    }
    .panel {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 18px;
        margin-bottom: 14px;
        box-shadow: 0 8px 20px rgba(16, 24, 40, 0.05);
    }
    .result-ok {
        border-left: 8px solid #16a34a;
    }
    .result-bad {
        border-left: 8px solid #dc2626;
    }
    .result-title {
        color: #111827;
        font-size: 1.8rem;
        font-weight: 800;
        margin-bottom: 4px;
    }
    .muted {
        color: #6b7280;
        font-size: 0.95rem;
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 14px 16px;
        box-shadow: 0 6px 16px rgba(16, 24, 40, 0.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def hero():
    st.markdown(
        """
        <div class="hero">
            <h1>Leather Defect Inspection</h1>
            <p>OpenVINO 모델로 가죽 이미지의 정상/불량 여부를 빠르게 판정합니다.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def is_lfs_pointer(path):
    if not path.exists():
        return False
    try:
        return path.read_bytes()[:80].startswith(b"version https://git-lfs.github.com/spec")
    except OSError:
        return False


def validate_model_files():
    missing = [path for path in (MODEL_XML, MODEL_BIN) if not path.exists()]
    if missing:
        st.error("모델 파일을 찾을 수 없습니다.")
        st.write("누락된 파일:", [str(path) for path in missing])
        st.stop()

    if is_lfs_pointer(MODEL_BIN):
        st.error("모델 가중치 파일이 Git LFS 포인터 상태입니다.")
        st.write("배포 환경에서 Git LFS 파일이 실제 모델 파일로 내려받아져야 합니다.")
        with st.expander("확인 정보"):
            st.write("모델 XML:", str(MODEL_XML))
            st.write("모델 BIN:", str(MODEL_BIN))
            st.write("BIN 크기:", f"{MODEL_BIN.stat().st_size:,} bytes")
            st.code("git lfs pull", language="bash")
        st.stop()


@st.cache_resource(show_spinner=False)
def load_model():
    validate_model_files()
    core = ov.Core()
    model = core.read_model(str(MODEL_XML))
    return core.compile_model(model, "CPU")


def detect_image_signature(image_bytes):
    header = image_bytes[:32]
    stripped = image_bytes[:100].lstrip().lower()

    if header.startswith(b"\xff\xd8\xff"):
        return "JPEG"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"
    if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "WEBP"
    if header[4:12] in {b"ftypavif", b"ftypheic", b"ftypheix", b"ftypmif1", b"ftypmsf1"}:
        return "AVIF/HEIC"
    if stripped.startswith((b"<!doctype html", b"<html")):
        return "HTML"
    return "알 수 없음"


def load_image(uploaded_file):
    image_bytes = uploaded_file.getvalue()
    if not image_bytes:
        st.error("이미지 파일이 비어 있습니다.")
        return None

    try:
        image = Image.open(BytesIO(image_bytes))
        image.load()
        return ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        detected = detect_image_signature(image_bytes)
        st.error("이미지를 읽을 수 없습니다. JPG 또는 PNG 파일을 업로드해주세요.")
        with st.expander("파일 정보"):
            st.write("파일명:", getattr(uploaded_file, "name", "카메라 입력"))
            st.write("브라우저 파일 형식:", getattr(uploaded_file, "type", "알 수 없음"))
            st.write("감지된 파일 내용:", detected)
            st.write("파일 크기:", f"{len(image_bytes):,} bytes")
            st.write("오류:", str(exc))
        return None


def preprocess(image):
    resized = image.convert("RGB").resize(INPUT_SIZE)
    arr = np.array(resized, dtype=np.float32)

    arr = arr[:, :, ::-1]
    arr[:, :, 0] -= 103.94
    arr[:, :, 1] -= 116.78
    arr[:, :, 2] -= 123.68

    return np.expand_dims(arr, axis=0)


def predict(compiled_model, image):
    arr = preprocess(image)
    result = compiled_model(arr)

    try:
        output = result[compiled_model.output(0)]
    except Exception:
        output = result[0]

    defect_probability = float(np.ravel(output)[0])
    return max(0.0, min(1.0, defect_probability))


def result_panel(defect_probability, threshold):
    normal_probability = 1.0 - defect_probability
    is_defect = defect_probability >= threshold
    label = CLASS_NAMES[1 if is_defect else 0]
    panel_class = "result-bad" if is_defect else "result-ok"
    subtitle = "불량 가능성이 기준값 이상입니다." if is_defect else "불량 가능성이 기준값보다 낮습니다."

    st.markdown(
        f"""
        <div class="panel {panel_class}">
            <div class="result-title">{label}</div>
            <div class="muted">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("정상 확률", f"{normal_probability:.1%}")
    col2.metric("불량 확률", f"{defect_probability:.1%}")
    col3.metric("판정 기준", f"{threshold:.0%}")
    st.progress(defect_probability)
    st.caption(f"불량 확률 {defect_probability:.1%}")


hero()

with st.sidebar:
    st.title("Inspection")
    threshold = st.slider("불량 판정 기준", 0.05, 0.95, 0.50, 0.05)
    st.divider()
    st.caption("입력 크기")
    st.write(f"{INPUT_SIZE[0]} x {INPUT_SIZE[1]}")
    st.caption("모델")
    st.write("OpenVINO CPU")
    st.caption("클래스")
    st.write("정상 / 불량")

try:
    model = load_model()
except Exception as exc:
    st.error("모델 로드 중 오류가 발생했습니다.")
    with st.expander("오류 상세"):
        st.exception(exc)
    st.stop()

input_col, result_col = st.columns([1.05, 0.95])

with input_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("이미지 입력")
    input_method = st.radio(
        "입력 방식",
        ["파일 업로드", "카메라 촬영"],
        horizontal=True,
    )

    selected_image = None
    if input_method == "파일 업로드":
        uploaded = st.file_uploader(
            "검사할 가죽 이미지를 업로드하세요",
            type=["jpg", "jpeg", "png"],
        )
        if uploaded is not None:
            selected_image = load_image(uploaded)
    else:
        captured = st.camera_input("카메라로 검사할 이미지를 촬영하세요")
        if captured is not None:
            selected_image = load_image(captured)

    if selected_image is not None:
        st.image(selected_image, caption="검사 대상 이미지", width=420)
    st.markdown("</div>", unsafe_allow_html=True)

with result_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("검사 결과")

    if selected_image is None:
        st.info("이미지를 입력하면 검사 결과가 표시됩니다.")
    else:
        if st.button("검사 시작", type="primary"):
            with st.spinner("이미지를 분석하는 중입니다."):
                probability = predict(model, selected_image)
            result_panel(probability, threshold)
        else:
            st.caption("검사 시작 버튼을 눌러 판정 결과를 확인하세요.")

    st.markdown("</div>", unsafe_allow_html=True)

st.caption("VGG16 전처리 기준으로 RGB 이미지를 224 x 224 크기로 변환한 뒤 OpenVINO 모델에 입력합니다.")
