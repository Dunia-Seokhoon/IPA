import os
import streamlit as st
import streamlit.components.v1 as components
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

load_dotenv()

# ─── API 키들 설정 ───────────────────────────────────────────────────────────
openai.api_key = (
    st.secrets.get("OPENAI_API_KEY")
    or os.getenv("OPENAI_API_KEY", "")
)
API_KEY      = os.getenv("ODCLOUD_API_KEY")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_API_URL   = os.getenv("HF_API_URL")

# ─── 1) 뉴스 크롤러 (Google News RSS) ─────────────────────────────────────────
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

# ─── 2) 선박 관제정보 조회 섹션 ────────────────────────────────────────────────
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

# ─── 3) 오늘의 날씨 섹션 ────────────────────────────────────────────────────────
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

        loc          = results[0]
        lat, lon     = loc["latitude"], loc["longitude"]
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

# ─── 4) Chatbot (Vision) & 요약 기능 ─────────────────────────────────────────
enc = tiktoken.encoding_for_model("gpt-4o-mini")
MAX_TOKENS = 131072  # gpt-4o-mini 최대 토큰 허용치 (약 131K)
SUMMARY_THRESHOLD = 40  # 대화 메시지(turn)가 40개 이상 넘어가면 요약

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
    if tk_in > MAX_TOKENS:
        raise ValueError(f"입력 토큰 {tk_in}개 → 최대 허용치({MAX_TOKENS}) 초과입니다.")
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
        img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()

def summarize_history(history: list) -> str:
    """
    전체 history(messages)를 짧게 요약하여 반환하는 함수.
    GPT에게 요약 요청을 보내고 3문장 이내 요약문을 받아옴.
    """
    prompt = [{"role": "system", "content": "아래 대화를 짧게 요약해 주세요."}]
    prompt += history + [{"role": "user", "content": "자, 이 대화 내용을 3문장 이내로 요약해 줘."}]
    res = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=prompt,
        max_tokens=200
    )
    return res.choices[0].message.content  # 요약된 텍스트

def chatgpt_clone_section():
    st.subheader("💬 Chatbot (Vision)")
    img_file = st.file_uploader("🖼️ 이미지 (선택)", type=["png", "jpg", "jpeg"])
    prompt   = st.chat_input("메시지를 입력하세요")

    # 세션 스테이트 초기화
    st.session_state.setdefault("chat_history", [])

    # 1) 대화 기록이 SUMMARY_THRESHOLD 턴을 초과하면 요약 수행
    if len(st.session_state.chat_history) > SUMMARY_THRESHOLD:
        try:
            summary = summarize_history(st.session_state.chat_history)
            # 요약된 텍스트를 assistant 역할로 저장 후, history를 재구성
            st.session_state.chat_history = [
                {"role": "assistant", "content": summary}
            ]
        except Exception as e:
            st.error(f"대화 요약 중 오류가 발생했습니다: {e}")
            return

    # 2) 화면에 이전 대화 내용 표시
    for msg in st.session_state.chat_history:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            with st.chat_message("user"):
                # content가 블록 리스트일 경우 처리
                if isinstance(content, list):
                    for blk in content:
                        if blk["type"] == "text":
                            st.write(blk["text"])
                        elif blk["type"] == "image_url":
                            st.image(blk["image_url"]["url"], caption="사용자 업로드 이미지")
                else:
                    st.write(content)
        else:  # assistant
            with st.chat_message("assistant"):
                st.write(content)

    # 3) 입력이 없으면 리턴
    if img_file is None and not prompt:
        return

    # 4) 사용자 입력 처리
    user_blocks = []
    if prompt:
        user_blocks.append({"type": "text", "text": prompt})
    if img_file:
        # compress_image 함수는 max_px=768, quality=85 고정
        jpg_bytes = compress_image(img_file)
        st.image(jpg_bytes, caption=f"미리보기 ({len(jpg_bytes)//1024} KB)", use_container_width=True)
        b64 = base64.b64encode(jpg_bytes).decode()
        img_block = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        user_blocks.append(img_block)

    # 5) 사용자 메시지를 chat_history에 추가
    st.session_state.chat_history.append({"role": "user", "content": user_blocks})

    # 6) 모델에게 보낼 메시지(prospective)를 구성 후 토큰 검증
    prospective = st.session_state.chat_history.copy()
    tk_in = num_tokens(prospective)
    if tk_in > MAX_TOKENS:
        st.error(f"현재 대화 토큰 수({tk_in})가 너무 많아 호출할 수 없습니다.\n"
                 "오래된 대화를 요약하거나 일부 메시지를 제거해 주세요.")
        return

    # 7) GPT 호출 및 출력
    try:
        resp = safe_chat_completion(st.session_state.chat_history)
        buf = ""
        # assistant 메시지를 미리 추가해두고, 내용을 스트리밍 중 계속 덧붙임
        st.session_state.chat_history.append({"role": "assistant", "content": ""})
        with st.chat_message("assistant"):
            ph = st.empty()
            for chunk in resp:
                delta = chunk.choices[0].delta.content
                if delta:
                    buf += delta
                    ph.markdown(buf + "▌")
            ph.markdown(buf)
        # 완성된 assistant 응답을 세션에 반영
        st.session_state.chat_history[-1]["content"] = buf
    except openai.RateLimitError:
        st.error("⏳ 레이트 리밋에 걸렸습니다. 잠시 후 다시 시도해 주세요.")
    except Exception as e:
        st.error(f"OpenAI 호출 오류: {e}")

# ─── 5) 영상 모음 섹션 ───────────────────────────────────────────────────────────
def video_collection_section():
    st.subheader("📺 ESG 영상 모음")
    # 1. 사무실에서 이면지 활용하기!
    st.markdown("#### 사무실에서 이면지 활용하기!")
    st.video("https://storage.googleapis.com/videoupload_icpa/%EC%82%AC%EB%AC%B4%EC%8B%A4%EC%97%90%EC%84%9C%20%EC%9D%B4%EB%A9%B4%EC%A7%80%20%ED%99%9C%EC%9A%A9%ED%95%98%EA%B8%B0.mp4")
    st.write("")  # 줄 간격

    # 2. 카페에서 ESG 실천하기 1탄
    st.markdown("#### 카페에서 ESG 실천하기 1탄")
    st.video("https://storage.googleapis.com/videoupload_icpa/%EC%B9%B4%ED%8E%98%EC%97%90%EC%84%9C%20%ED%85%80%EB%B8%94%EB%9F%AC%20%EC%82%AC%EC%9A%A9%ED%95%98%EA%B8%B0.mp4")
    st.write("")

    # 3. 카페에서 휴지 적게 사용하기
    st.markdown("#### 카페에서 휴지 적게 사용하기")
    st.video("https://storage.googleapis.com/videoupload_icpa/%EC%B9%B4%ED%8E%98%EC%97%90%EC%84%9C%20%ED%9C%B4%EC%A7%80%20%EC%A0%81%EA%B2%8C%20%EC%82%AC%EC%9A%A9%ED%95%98%EA%B8%B0.mp4")

# ─── 6) 앱 레이아웃 (탭 구성) ─────────────────────────────────────────────────────
st.set_page_config(page_title="통합 데모", layout="centered")
st.title("📈 통합 데모: 뉴스·선박·날씨·Chatbot·영상 모음")

tabs = st.tabs([
    "구글 뉴스", "선박 관제정보", "오늘의 날씨", "Chatbot", "ESG 영상 모음"
])

with tabs[0]:
    st.subheader("▶ 구글 뉴스 크롤링 (RSS)")
    kw  = st.text_input("검색 키워드", "ESG")
    num = st.slider("가져올 기사 개수", 5, 50, 10)
    if st.button("보기", key="news_btn"):
        for it in fetch_google_news(kw, num):
            st.markdown(f"- **[{it['source']} · {it['date']}]** [{it['title']}]({it['link']})")

with tabs[1]:
    vessel_monitoring_section()

with tabs[2]:
    today_weather_section()

with tabs[3]:
    chatgpt_clone_section()

with tabs[4]:
    video_collection_section()













