"""
tf_App_openvino.py
──────────────────
OpenVINO 모델(.xml / .bin)로 이진분류 추론하는 기본 구조 예제.

[전체 흐름]
  1. 페이지 설정
  2. 모델 로드        : OpenVINO Core → compile_model
  3. 이미지 전처리    : PIL → numpy → VGG16 정규화
  4. 추론             : compiled_model(img_array)
  5. 결과 표시        : 정상 / 불량 + 확률
"""

import streamlit as st
import numpy as np
from io import BytesIO
from pathlib import Path
from PIL import Image, ImageOps, UnidentifiedImageError
import openvino as ov

# ── 상수 ──────────────────────────────────────
INPUT_IMG_SIZE = (224, 224)
CLASSES        = ["정상", "불량"]
APP_DIR        = Path(__file__).resolve().parent
MODEL_DIR      = APP_DIR / "weights"
MODEL_XML      = MODEL_DIR / "leather_model.xml"
MODEL_BIN      = MODEL_XML.with_suffix(".bin")


# ─────────────────────────────────────────────
# 1. 페이지 설정
# ─────────────────────────────────────────────
st.set_page_config(page_title="InspectorsAlly", page_icon=":camera:", layout="centered")
st.title("가죽이 이상해!")
st.caption("VGG16 + OpenVINO 기반 가죽 제품 불량 검사")


# ─────────────────────────────────────────────
# 2. 모델 로드
#    ov.Core()       : OpenVINO 런타임 초기화
#    read_model()    : .xml 구조 + .bin 가중치 읽기
#    compile_model() : CPU 추론 엔진으로 컴파일
#
#    @st.cache_resource : 앱 재실행 시 모델을 다시 로드하지 않음
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    core  = ov.Core()
    model = core.read_model(str(MODEL_XML))
    return core.compile_model(model, "CPU")


def show_model_path_debug(missing_files):
    st.error("모델 파일을 찾을 수 없습니다.")
    st.write("누락된 파일:", [str(path) for path in missing_files])

    with st.expander("경로 디버그 정보"):
        st.write("현재 작업 경로:", str(Path.cwd()))
        st.write("현재 파일 위치:", str(Path(__file__).resolve()))
        st.write("모델 폴더:", str(MODEL_DIR))
        st.write("weights 존재:", MODEL_DIR.exists())
        st.write(
            "weights 내부:",
            [path.name for path in MODEL_DIR.glob("*")] if MODEL_DIR.exists() else "없음",
        )


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


def format_header_bytes(image_bytes, length=16):
    return " ".join(f"{byte:02X}" for byte in image_bytes[:length])


def load_input_image(image_file):
    image_bytes = image_file.getvalue()
    if not image_bytes:
        st.error("이미지 파일이 비어 있습니다. 다른 파일을 선택해주세요.")
        return None

    try:
        image = Image.open(BytesIO(image_bytes))
        image.load()
        return ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        detected_format = detect_image_signature(image_bytes)

        if detected_format == "HTML":
            st.error("업로드한 파일 내용이 이미지가 아니라 웹페이지(HTML)로 보입니다. 이미지를 다시 다운로드해주세요.")
        elif detected_format == "AVIF/HEIC":
            st.error("이 파일은 JPG가 아니라 AVIF/HEIC 계열로 보입니다. JPG 또는 PNG로 변환한 뒤 업로드해주세요.")
        elif detected_format == "WEBP":
            st.error("이 파일은 JPG가 아니라 WebP로 보입니다. JPG 또는 PNG로 변환한 뒤 업로드해주세요.")
        else:
            st.error("이미지를 읽을 수 없습니다. JPG, JPEG, PNG 파일로 다시 저장한 뒤 업로드해주세요.")

        with st.expander("파일 정보"):
            st.write("파일명:", getattr(image_file, "name", "카메라 입력"))
            st.write("브라우저가 보낸 파일 형식:", getattr(image_file, "type", "알 수 없음"))
            st.write("감지된 파일 내용:", detected_format)
            st.write("파일 크기:", f"{len(image_bytes):,} bytes")
            st.write("파일 첫 16바이트:", format_header_bytes(image_bytes))
            st.write("오류:", str(exc))
        return None


# ─────────────────────────────────────────────
# 3. 이미지 전처리
#    VGG16 훈련 시 사용한 전처리와 동일하게 맞춰야 예측이 정확함
#
#    ① RGB → BGR 채널 순서 변환  (VGG16은 BGR 입력)
#    ② ImageNet BGR 평균값 빼기
#       B: 103.94 / G: 116.78 / R: 123.68
#    ③ 배치 차원 추가 (224,224,3) → (1,224,224,3)
# ─────────────────────────────────────────────
def preprocess(pil_img):
    img   = pil_img.convert("RGB").resize(INPUT_IMG_SIZE)
    arr   = np.array(img, dtype=np.float32)

    arr   = arr[:, :, ::-1]        # RGB → BGR
    arr[:, :, 0] -= 103.94         # B 채널 평균 빼기
    arr[:, :, 1] -= 116.78         # G 채널 평균 빼기
    arr[:, :, 2] -= 123.68         # R 채널 평균 빼기

    return np.expand_dims(arr, axis=0)   # (1, 224, 224, 3)


# ─────────────────────────────────────────────
# 4. 추론
#    compiled_model(input) → OVDict
#    결과는 인덱스[0] 으로 접근: shape (1, 1)
#    sigmoid 출력값 → 0에 가까울수록 정상, 1에 가까울수록 불량
# ─────────────────────────────────────────────
def predict(compiled_model, pil_img):
    arr  = preprocess(pil_img)
    prob = float(compiled_model(arr)[0][0][0])   # sigmoid 단일값
    label = CLASSES[1 if prob > 0.5 else 0]
    return label, prob


# ─────────────────────────────────────────────
# 5. 메인 UI
# ─────────────────────────────────────────────
missing_model_files = [path for path in (MODEL_XML, MODEL_BIN) if not path.exists()]
if missing_model_files:
    show_model_path_debug(missing_model_files)
    st.stop()

try:
    compiled_model = load_model()
except Exception as exc:
    st.error("모델 로드 중 오류가 발생했습니다.")
    with st.expander("오류 상세"):
        st.exception(exc)
    st.stop()

# ── 입력 방법 선택 ──
st.subheader("이미지 입력")
input_method = st.radio("options", ["파일 업로드", "카메라 촬영"],
                        label_visibility="collapsed")
pil_image = None

if input_method == "파일 업로드":
    file = st.file_uploader("이미지를 선택하세요", type=["jpg", "jpeg", "png"])
    if file:
        pil_image = load_input_image(file)
        if pil_image is not None:
            st.image(pil_image, caption="업로드된 이미지", width=300)

elif input_method == "카메라 촬영":
    shot = st.camera_input("카메라로 촬영")
    if shot:
        pil_image = load_input_image(shot)
        if pil_image is not None:
            st.image(pil_image, caption="촬영된 이미지", width=300)

# ── 검사 버튼 ──
if st.button("검사 시작", type="primary"):
    if pil_image is None:
        st.warning("이미지를 먼저 입력해주세요.")
    else:
        with st.spinner("분석 중..."):
            label, prob = predict(compiled_model, pil_image)

        # ── 결과 표시 ──
        st.subheader("검사 결과")

        if label == "정상":
            st.success(f"✅ **정상**  (불량 확률: {prob:.1%})")
        else:
            st.error(f"⚠️ **불량 감지**  (불량 확률: {prob:.1%})")

        col1, col2 = st.columns(2)
        col1.metric("정상 확률", f"{1 - prob:.1%}")
        col2.metric("불량 확률", f"{prob:.1%}")
        st.progress(float(prob), text=f"불량 확률: {prob:.1%}")
