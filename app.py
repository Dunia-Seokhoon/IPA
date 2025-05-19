# app.py  |  Streamlit 통합 데모 (HF Llama3 PDF 챗봇)

import os
from datetime import datetime, date
from urllib.parse import urlencode, quote_plus

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import feedparser
import requests

# ───────────── LangChain × Hugging Face ─────────────
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceHubEmbeddings,
)
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

# ───────────── Secrets ─────────────
API_KEY    = os.getenv("ODCLOUD_API_KEY")             # 해양수산부 오픈데이터
HF_API_TOKEN = os.getenv("HF_API_TOKEN")              # GPT-2 테스트용
HFHUB_TOKEN  = os.getenv("HUGGINGFACEHUB_API_TOKEN")  # Llama3·임베딩용

# ────────────────── 1) 구글 뉴스 ──────────────────
@st.cache_data(ttl=300)
def fetch_google_news(keyword: str, max_items: int = 10):
    params  = {"q": keyword, "hl": "ko", "gl": "KR", "ceid": "KR:ko"}
    rss_url = "https://news.google.com/rss/search?" + urlencode(params, doseq=True)
    feed    = feedparser.parse(rss_url)
    items   = []
    for e in feed.entries[:max_items]:
        d = datetime(*e.published_parsed[:6]).strftime("%Y-%m-%d")
        items.append({"title": e.title, "link": e.link,
                      "source": e.get("source", {}).get("title", ""), "date": d})
    return items

# ────────────────── 2) CSV 히스토그램 ──────────────────
def sample_data_section():
    st.subheader("📊 샘플 데이터 히스토그램")
    upl = st.file_uploader("CSV 파일 업로드 (optional)", type=["csv"])
    if not upl:
        st.info("CSV 파일을 올리면 히스토그램을 볼 수 있습니다.")
        return
    df = pd.read_csv(upl)
    st.dataframe(df)
    nums = df.select_dtypes(include="number").columns.tolist()
    if not nums:
        st.warning("숫자형 컬럼이 없습니다.")
        return
    col = st.selectbox("Numeric 컬럼 선택", nums)
    fig, ax = plt.subplots()
    ax.hist(df[col], bins=10)
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

# ────────────────── 3) 동영상 재생 ──────────────────
def video_upload_section():
    st.subheader("📹 동영상 업로드 & 재생")
    vfile = st.file_uploader("동영상 파일 업로드", type=["mp4", "mov", "avi"])
    st.video(vfile) if vfile else st.info("파일을 선택해 주세요.")

# ────────────────── 4) 선박 관제정보 ──────────────────
def vessel_monitoring_section():
    st.subheader("🚢 해양수산부 선박 관제정보 조회")
    d_from = st.date_input("조회 시작일", date.today())
    d_to   = st.date_input("조회 종료일", date.today())
    page   = st.number_input("페이지 번호", 1, 1000, 1)
    per    = st.slider("한 번에 가져올 건수", 1, 1000, 100)
    if not st.button("🔍 조회"):
        return
    params = {
        "serviceKey": API_KEY, "page": page, "perPage": per,
        "fromDate": d_from.strftime("%Y-%m-%d"), "toDate": d_to.strftime("%Y-%m-%d"),
    }
    with st.spinner("조회 중…"):
        r = requests.get(
            "https://api.odcloud.kr/api/15128156/v1/uddi:fdcdb0d1-0296-4c3b-8087-8ab4bd4d5123",
            params=params)
    if r.status_code != 200:
        st.error(f"API 오류 {r.status_code}")
        return
    data = r.json().get("data", [])
    if data:
        st.success(f"총 {len(data)} 건 조회되었습니다.")
        st.dataframe(pd.DataFrame(data))
    else:
        st.warning("조회된 데이터가 없습니다.")

# ────────────────── 5) 오늘의 날씨 ──────────────────
def today_weather_section():
    st.subheader("☀️ 오늘의 날씨 조회")
    city = st.text_input("도시 이름 입력 (예: 서울, Busan)")
    if not st.button("🔍 날씨 가져오기"):
        return
    if not city:
        st.warning("도시 이름을 입력해 주세요."); return
    q     = quote_plus(city)
    geo_r = requests.get(
        f"https://geocoding-api.open-meteo.com/v1/search?name={q}&count=5&language=ko")
    if geo_r.status_code != 200:
        st.error("지오코딩 API 오류"); return
    j = geo_r.json().get("results")
    if not j:
        st.warning("도시를 찾을 수 없습니다."); return
    loc = j[0]; lat, lon = loc["latitude"], loc["longitude"]
    w_r = requests.get(
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&current_weather=true"
        f"&hourly=relativehumidity_2m&timezone=auto")
    if w_r.status_code != 200:
        st.error("날씨 API 오류"); return
    cw  = w_r.json()["current_weather"]
    wc  = {0:"맑음",1:"주로 맑음",2:"부분적 구름",3:"구름 많음",
           45:"안개",48:"안개(입상)",51:"이슬비 약함",53:"이슬비 보통",
           55:"이슬비 강함",61:"빗방울 약함",63:"빗방울 보통",65:"빗방울 강함",
           80:"소나기 약함",81:"소나기 보통",82:"소나기 강함",
           95:"뇌우",96:"약한 뇌우",99:"강한 뇌우"}
    st.markdown(f"### {loc['name']}, {loc['country']} 현재 날씨")
    c1,c2,c3 = st.columns(3)
    c1.metric("🌡️ 기온(℃)", cw["temperature"])
    c2.metric("💨 풍속(m/s)", cw["windspeed"])
    c3.metric("상태", wc.get(cw["weathercode"], "알 수 없음"))

# ────────────────── 6) GPT-2 테스트 ──────────────────
def generate_with_gpt2(prompt: str) -> str:
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    r = requests.post(
        "https://api-inference.huggingface.co/models/gpt2",
        headers=headers,
        json={"inputs":prompt,"parameters":{"max_new_tokens":150}},
        timeout=30)
    r.raise_for_status()
    return r.json()[0]["generated_text"]

def llm_section():
    st.subheader("🤖 GPT-2 테스트 (HF Inference API)")
    p = st.text_area("프롬프트 입력", height=150)
    if st.button("생성"):
        try:
            with st.spinner("생성 중…"):
                st.write(generate_with_gpt2(p))
        except Exception as e:
            st.error(e)

# ────────────────── 7) PDF 챗봇 (HF Llama-3 8B) ──────────────────
def pdf_chatbot_section():
    st.subheader("📑 PDF 챗봇 (Llama-3 8B + MiniLM)")
    pdf = st.file_uploader("PDF 업로드", type=["pdf"])
    if "hist" not in st.session_state:
        st.session_state.hist = []
    if not pdf:
        st.info("PDF 파일을 올리면 질문할 수 있습니다."); return

    # 1) 문서 로드 & 분할
    docs = PyPDFLoader(pdf).load_and_split()

    # 2) 임베딩 & 벡터스토어
    embed = HuggingFaceHubEmbeddings(
        model_name             ="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=HFHUB_TOKEN)
    store = FAISS.from_documents(docs, embed)

    # 3) 챗 모델
    model = ChatHuggingFace(
        repo_id                ="meta-llama/Meta-Llama-3-8B-Instruct",
        huggingfacehub_api_token=HFHUB_TOKEN,
        temperature            =0.2)

    chain = ConversationalRetrievalChain.from_llm(
        llm       =model,
        retriever =store.as_retriever(),
        return_source_documents=True)

    q = st.text_input("질문을 입력하세요")
    if st.button("질문하기") and q:
        with st.spinner("답변 생성 중…"):
            res = chain({"question":q,"chat_history":st.session_state.hist})
        ans = res["answer"]
        st.session_state.hist.append((q, ans))
        st.markdown("### 💬 답변")
        st.write(ans)
        with st.expander("🔍 참조 문서"):
            for d in res["source_documents"]:
                pnum = d.metadata.get("page","?")
                st.markdown(f"- p.{pnum}: {d.page_content[:120]}…")

# ────────────────── 8) 페이지 레이아웃 ──────────────────
st.set_page_config(page_title="통합 데모", layout="centered")
st.title("📈 통합 데모: 뉴스·데이터·동영상·선박·날씨·LLM·PDF 챗봇")

tabs = st.tabs([
    "구글 뉴스", "데이터 히스토그램", "동영상 재생",
    "선박 관제정보", "오늘의 날씨", "LLM 테스트", "PDF 챗봇"
])

with tabs[0]:  # 뉴스
    st.subheader("▶ 구글 뉴스 (RSS)")
    kw  = st.text_input("검색 키워드", "ESG")
    num = st.slider("가져올 기사 개수", 5, 20, 10)
    if st.button("뉴스 보기"):
        for it in fetch_google_news(kw, num):
            st.markdown(f"- **[{it['source']} · {it['date']}]** [{it['title']}]({it['link']})")

with tabs[1]: sample_data_section()
with tabs[2]: video_upload_section()
with tabs[3]: vessel_monitoring_section()
with tabs[4]: today_weather_section()
with tabs[5]: llm_section()
with tabs[6]: pdf_chatbot_section()





