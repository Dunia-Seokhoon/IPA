import os
import streamlit as st
import openai
import base64
import backoff
import tiktoken
import time
import pandas as pd
import matplotlib.pyplot as plt
import feedparser
import requests
from datetime import datetime, date
from urllib.parse import urlencode, quote_plus
from dotenv import load_dotenv
from llama_index.core import (
    VectorStoreIndex, SimpleDirectoryReader,
    StorageContext, load_index_from_storage, Settings
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
import logging, traceback
from io import BytesIO
from PIL import Image

# API 키들 설정
openai.api_key = (
    st.secrets.get("OPENAI_API_KEY")
    or os.getenv("OPENAI_API_KEY", "")
)
API_KEY      = os.getenv("ODCLOUD_API_KEY")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_API_URL   = os.getenv("HF_API_URL")

# ────────────────── 1) 뉴스 크롤러 (Google News RSS) ──────────────────
@st.cache_data(ttl=300)
def fetch_google_news(keyword: str, max_items: int = 10):
    clean_kw = " ".join(keyword.strip().split())
    params = {"q": clean_kw, "hl": "ko", "gl": "KR", "ceid": "KR:ko"}
    rss_url = "https://news.google.com/rss/search?" + urlencode(params, doseq=True)
    feed = feedparser.parse(rss_url)
    items = []
    for entry in feed.entries[:max_items]:
        pub_date = datetime(*entry.published_parsed[:6]).strftime("%Y-%m-%d")
        source   = entry.get("source", {}).get("title", "")
        items.append({
            "title":  entry.title,
            "link":   entry.link,
            "source": source,
            "date":   pub_date,
        })
    return items

# ────────────────── 2) CSV 히스토그램 섹션 ──────────────────
def sample_data_section():
    st.subheader("📊 샘플 데이터 히스토그램")
    uploaded_file = st.file_uploader("CSV 파일 업로드 (optional)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df)
        nums = df.select_dtypes(include="number").columns.tolist()
        if nums:
            col = st.selectbox("Numeric 컬럼 선택", nums)
            fig, ax = plt.subplots()
            ax.hist(df[col], bins=10)
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
    else:
        st.info("CSV 파일을 올리면 히스토그램을 볼 수 있습니다.")

# ────────────────── 3) 동영상 업로드·재생 섹션 ──────────────────
def video_upload_section():
    st.subheader("📹 동영상 업로드 & 재생")
    video_file = st.file_uploader("동영상 파일 업로드", type=["mp4","mov","avi"])
    if video_file:
        st.video(video_file)
    else:
        st.info("파일을 선택해 주세요.")

# ────────────────── 4) 선박 관제정보 조회 섹션 ──────────────────
def vessel_monitoring_section():
    st.subheader("🚢 해양수산부 선박 관제정보 조회")
    date_from = st.date_input("조회 시작일", date.today())
    date_to   = st.date_input("조회 종료일", date.today())
    page      = st.number_input("페이지 번호", 1, 1000, 1)
    per_page  = st.slider("한 번에 가져올 건수", 1, 1000, 100)
    if st.button("🔍 조회"):
        params = {
            "serviceKey": API_KEY,
            "page":       page,
            "perPage":    per_page,
            "fromDate":   date_from.strftime("%Y-%m-%d"),
            "toDate":     date_to.strftime("%Y-%m-%d"),
        }
        with st.spinner("조회 중…"):
            res = requests.get(
                "https://api.odcloud.kr/api/15128156/v1/uddi:fdcdb0d1-0296-4c3b-8087-8ab4bd4d5123",
                params=params
            )
        if res.status_code != 200:
            st.error(f"API 오류 {res.status_code}")
            return
        data = res.json().get("data", [])
        if data:
            df = pd.DataFrame(data)
            st.success(f"총 {len(df)} 건 조회되었습니다.")
            st.dataframe(df)
        else:
            st.warning("조회된 데이터가 없습니다.")

# ────────────────── 5) 오늘의 날씨 섹션 ──────────────────
def today_weather_section():
    st.subheader("☀️ 오늘의 날씨 조회")
    city_name = st.text_input("도시 이름 입력 (예: 서울, Busan)")
    if st.button("🔍 날씨 가져오기"):
        if not city_name:
            st.warning("도시 이름을 입력해 주세요.")
            return

        q_name  = quote_plus(city_name)
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={q_name}&count=5&language=ko"
        with st.spinner("위치 검색 중…"):
            geo_res = requests.get(geo_url)
        if geo_res.status_code != 200:
            st.error("지오코딩 API 오류")
            return
        results = geo_res.json().get("results")
        if not results:
            st.warning("도시를 찾을 수 없습니다.")
            return

        loc  = results[0]
        lat  = loc["latitude"]
        lon  = loc["longitude"]
        display_name = f"{loc['name']}, {loc['country']}"

        weather_url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}&current_weather=true"
            f"&hourly=relativehumidity_2m&timezone=auto"
        )
        with st.spinner(f"{display_name} 날씨 불러오는 중…"):
            w_res = requests.get(weather_url)
        if w_res.status_code != 200:
            st.error("날씨 API 오류")
            return

        js   = w_res.json()
        cw   = js.get("current_weather", {})
        temp, wind_spd, wind_dir, code = (
            cw.get("temperature"),
            cw.get("windspeed"),
            cw.get("winddirection"),
            cw.get("weathercode"),
        )
        wc_map = {
            0:"맑음",1:"주로 맑음",2:"부분적 구름",3:"구름 많음",
            45:"안개",48:"안개(입상)",
            51:"이슬비 약함",53:"이슬비 보통",55:"이슬비 강함",
            61:"빗방울 약함",63:"빗방울 보통",65:"빗방울 강함",
            80:"소나기 약함",81:"소나기 보통",82:"소나기 강함",
            95:"뇌우",96:"약한 뇌우",99:"강한 뇌우"
        }
        desc      = wc_map.get(code, "알 수 없음")
        times     = js["hourly"]["time"]
        hums      = js["hourly"]["relativehumidity_2m"]
        now       = datetime.now().strftime("%Y-%m-%dT%H:00")
        humidity  = hums[times.index(now)] if now in times else None

        st.markdown(f"### {display_name} 현재 날씨")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🌡️ 기온(℃)", temp)
        c2.metric("💨 풍속(m/s)", wind_spd)
        c3.metric("🌫️ 풍향(°)", wind_dir)
        c4.metric("💧 습도(%)", humidity or "–")
        st.markdown(f"**날씨 상태:** {desc}")

# ────────────────── 6) LLM 테스트 (Hugging Face Inference API) ──────────────────
def generate_with_kanana(prompt: str) -> str:
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {
        "inputs":     prompt,
        "options":    {"use_cache": False},
        "parameters": {"max_new_tokens": 150, "temperature": 0.7}
    }
    res = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)
    res.raise_for_status()
    return res.json()[0]["generated_text"]

def llm_section():
    st.subheader("🤖 카나나 Nano (Hugging Face Inference API)")
    prompt = st.text_area("프롬프트 입력", height=150)
    if st.button("생성"):
        if not HF_API_TOKEN or not HF_API_URL:
            st.error("HF_API_TOKEN 또는 HF_API_URL이 설정되지 않았습니다.")
            return
        with st.spinner("응답 생성 중…"):
            try:
                out = generate_with_kanana(prompt)
                st.markdown("### 응답")
                st.write(out)
            except Exception as e:
                st.error(f"LLM 호출 오류: {e}")

# ────────────────── 7) 문서 기반 챗봇 (LlamaIndex) ──────────────────
def rag_chatbot_section():
    st.subheader("📚 문서 기반 챗봇 (RAG with LlamaIndex)")

    with st.sidebar:
        st.markdown("### 🔑 OpenAI API Key")
        api_key = st.text_input(
            "OPENAI_API_KEY",
            value=st.secrets.get("OPENAI_API_KEY", ""),
            type="password"
        )
        uploaded_file = st.file_uploader(
            "📄 인덱싱할 문서 업로드",
            type=["txt", "pdf", "md", "docx", "pptx", "csv"]
        )

    if "rag_messages" not in st.session_state:
        st.session_state.rag_messages = []
    if "chat_engine" not in st.session_state:
        st.session_state.chat_engine = None

    os.makedirs("./cache/data", exist_ok=True)
    os.makedirs("./storage",    exist_ok=True)

    if uploaded_file is not None:
        file_path = os.path.join("cache", "data", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"‘{uploaded_file.name}’ 업로드 완료!")

    if api_key:
        load_dotenv()
        Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
        Settings.embed_model = OpenAIEmbedding(
            model="text-embedding-3-small",
            api_key=api_key
        )
    else:
        st.warning("🔑 OpenAI API Key를 입력해 주세요.")
        return

    @st.cache_resource(show_spinner="🔧 인덱스 빌드 중…")
    def load_or_build_index() -> VectorStoreIndex:
        persist_dir = "./storage"
        if os.listdir("cache/data"):
            docs = SimpleDirectoryReader("cache/data").load_data()
            idx  = VectorStoreIndex.from_documents(docs)
            idx.storage_context.persist(persist_dir)
            return idx
        if os.path.exists(os.path.join(persist_dir, "docstore.json")):
            sc  = StorageContext.from_defaults(persist_dir=persist_dir)
            return load_index_from_storage(sc)
        return None

    index = load_or_build_index()
    if index is None:
        st.info("먼저 문서를 업로드하거나 storage 폴더에 기존 인덱스를 두세요.")
        return

    if st.session_state.chat_engine is None:
        st.session_state.chat_engine = index.as_chat_engine(
            chat_mode="context",
            similarity_top_k=4,
            streaming=True
        )

    for msg in st.session_state.rag_messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    user_input = st.chat_input("질문을 입력하세요.", key="rag_input")
    if user_input:
        st.session_state.rag_messages.append({"role":"user","content":user_input})
        st.chat_message("user").markdown(user_input)
        try:
            with st.chat_message("assistant"):
                stream_resp = st.session_state.chat_engine.stream_chat(user_input)
                buf = ""
                for chunk in stream_resp.response_gen:
                    buf += chunk
                    st.write(buf + "▌")
                st.session_state.rag_messages.append({"role":"assistant","content":buf})
        except Exception as e:
            st.error(f"⚠️ 오류: {e}")
            traceback.print_exc()

# ────────────────── ChatGPT 클론 (Vision) 섹션 ──────────────────
enc = tiktoken.encoding_for_model("gpt-4o-mini")

def num_tokens(messages: list) -> int:
    total = 0
    for m in messages:
        if isinstance(m["content"], list):
            for blk in m["content"]:
                if blk["type"] == "text":
                    total += len(enc.encode(blk["text"]))
                elif blk["type"] == "image_url":
                    total += len(enc.encode(blk["image_url"]["url"]))
        else:
            total += len(enc.encode(m["content"]))
    return total

@backoff.on_exception(backoff.expo, openai.RateLimitError, max_time=60, jitter=None)
def safe_chat_completion(messages, model="gpt-4o-mini"):
    tk_in = num_tokens(messages)
    if tk_in > 50_000:
        raise ValueError(f"입력 토큰 {tk_in}개 → 너무 큽니다. 프롬프트/이미지 크기를 줄여주세요.")
    return openai.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=300,
        stream=True
    )

def compress_image(file, max_px=768, quality=85):
    img = Image.open(file)
    if max(img.size) > max_px:
        ratio = max_px / max(img.size)
        img = img.resize((int(img.width*ratio), int(img.height*ratio)), Image.LANCZOS)
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()

def chatgpt_clone_section():
    st.subheader("💬 ChatGPT 클론 (Vision)")
    img_file = st.file_uploader("🖼️ 이미지 (선택)", type=["png","jpg","jpeg"])
    col1, col2 = st.columns(2)
    max_px   = col1.slider("최대 해상도(px)", 256, 1024, 768, 128)
    quality  = col2.slider("JPEG 품질(%)", 30, 95, 85, 5)
    prompt   = st.chat_input("메시지를 입력하세요")

    if img_file is None and not prompt:
        return

    user_blocks = []
    if prompt:
        user_blocks.append({"type":"text", "text":prompt})
    if img_file:
        jpg_bytes = compress_image(img_file, max_px, quality)
        st.image(jpg_bytes, caption=f"미리보기 ({len(jpg_bytes)//1024} KB)", use_container_width=True)
        b64 = base64.b64encode(jpg_bytes).decode()
        img_block = {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}}
        user_blocks.append(img_block)

    prospective = st.session_state.get("gpt_msgs", []) + [{"role":"user","content":user_blocks}]
    if num_tokens(prospective) > 50_000:
        st.error("⚠️ 토큰 수 제한 초과. 해상도/품질을 줄여 주세요.")
        return

    st.session_state.setdefault("gpt_msgs", [])
    st.session_state.gpt_msgs.append({"role":"user","content":user_blocks})

    try:
        resp = safe_chat_completion(st.session_state.gpt_msgs)
        buf = ""
        with st.chat_message("assistant"):
            ph = st.empty()
            for chunk in resp:
                delta = chunk.choices[0].delta.content
                if delta:
                    buf += delta
                    ph.markdown(buf + "▌")
            ph.markdown(buf)
        st.session_state.gpt_msgs.append({"role":"assistant","content":buf})
    except openai.RateLimitError:
        st.error("⏳ 레이트 리밋에 걸렸습니다. 잠시 뒤 다시 시도해 주세요.")
    except Exception as e:
        st.error(f"OpenAI 호출 오류: {e}")

# ────────────────── 8) 앱 레이아웃 (탭 구성) ──────────────────
st.set_page_config(page_title="통합 데모", layout="centered")
st.title("📈 통합 데모: 뉴스·데이터·동영상·선박·날씨·LLM")

tabs = st.tabs([
    "구글 뉴스", "데이터 히스토그램", "동영상 재생",
    "선박 관제정보", "오늘의 날씨", "LLM 테스트", "문서 챗봇", "ChatGPT 클론",
    "유튜브 링크"
])

with tabs[0]:
    st.subheader("▶ 구글 뉴스 크롤링 (RSS)")
    kw  = st.text_input("검색 키워드", "ESG")
    num = st.slider("가져올 기사 개수", 5, 50, 10)
    if st.button("보기"):
        for it in fetch_google_news(kw, num):
            st.markdown(f"- **[{it['source']} · {it['date']}]** [{it['title']}]({it['link']})")

with tabs[1]:
    sample_data_section()

with tabs[2]:
    video_upload_section()

with tabs[3]:
    vessel_monitoring_section()

with tabs[4]:
    today_weather_section()

with tabs[5]:
    llm_section()

with tabs[6]:
    rag_chatbot_section()

with tabs[7]:
    chatgpt_clone_section()

with tabs[8]:
    st.subheader("📺 유튜브 동영상 임베드")
    yt_url = "https://www.youtube.com/watch?v=C7rRKxsqCk4&list=PLMojrPlCX93sjjUH3QQLi0mCYuGfJSfXH&index=12"
    st.video(yt_url)
    # 또는 iframe 제어가 필요할 때:
    # import streamlit.components.v1 as components
    # embed_url = "https://www.youtube.com/embed/C7rRKxsqCk4?list=PLMojrPlCX93sjjUH3QQLi0mCYuGfJSfXH&index=12"
    # components.iframe(embed_url, width=700, height=400)






