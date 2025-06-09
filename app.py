# -*- coding: utf-8 -*-
# ──────────────────────────────────────────────────────────────────────────────
#  인천항만공사 ESG 통합 포털  (뉴스 · 선박 · 날씨 · Chatbot · 댓글 · 활동참여 · 영상)
# ──────────────────────────────────────────────────────────────────────────────
import os                         # ← 오타(mport) 수정
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

# ─── 환경변수 & API 키 ─────────────────────────────────────────────────────────
load_dotenv()
openai.api_key = (
    st.secrets.get("OPENAI_API_KEY")
    or os.getenv("OPENAI_API_KEY", "")
)
API_KEY      = os.getenv("ODCLOUD_API_KEY")      # 선박 관제 API
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_API_URL   = os.getenv("HF_API_URL")

# ──────────────────────────────────────────────────────────────────────────────
# 1) Google News RSS  +  링크 일괄 복사
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_google_news(keyword: str, max_items: int = 10):
    clean_kw = " ".join(keyword.strip().split())
    params   = {"q": clean_kw, "hl": "ko", "gl": "KR", "ceid": "KR:ko"}
    rss_url  = "https://news.google.com/rss/search?" + urlencode(params, doseq=True)
    feed     = feedparser.parse(rss_url)
    items    = []
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

def google_news_section():
    st.subheader("📰 Google News 검색")
    kw        = st.text_input("키워드", value="카카오")
    max_items = st.slider("가져올 기사 개수", 5, 100, 10)
    if st.button("보기", key="news_search_btn"):
        news = fetch_google_news(kw, max_items)
        if not news:
            st.info("검색 결과가 없습니다.")
            return

        # ① 결과 목록
        for item in news:
            st.markdown(
                f"- **[{item['source']}] · {item['date']}** "
                f"[{item['title']}]({item['link']})",
                unsafe_allow_html=True
            )

        # ② 링크 일괄 복사용 문자열
        links_str = "\n".join([n["link"] for n in news])
        st.text_area("🔗 링크 일괄 복사용", links_str, height=100)

        # ③ copy-to-clipboard 버튼 (JS)
        btn_html = f"""
        <button id="copy-btn"
                style="margin-top:6px;padding:6px 12px;background:#f44336;
                       color:white;border:none;border-radius:4px;cursor:pointer;"
                onclick="navigator.clipboard.writeText(`{links_str}`);
                         const t=this.innerText; this.innerText='✅ 복사 완료!';
                         setTimeout(()=>this.innerText=t,1500);">
            📋 {len(news)}개 링크 복사
        </button>
        """
        st.markdown(btn_html, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# 2) 해양수산부 선박 관제 정보
# ──────────────────────────────────────────────────────────────────────────────
def vessel_monitoring_section():
    st.subheader("🚢 해양수산부 선박 관제정보 조회")
    date_from = st.date_input("조회 시작일", date.today())
    date_to   = st.date_input("조회 종료일", date.today())
    page      = st.number_input("페이지 번호", 1, 1000, 1)
    per_page  = st.slider("한 번에 가져올 건수", 1, 1000, 100)
    if st.button("🔍 조회", key="vessel_btn"):
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

# ──────────────────────────────────────────────────────────────────────────────
# 3) 오늘의 날씨
# ──────────────────────────────────────────────────────────────────────────────
def today_weather_section():
    st.subheader("☀️ 오늘의 날씨 조회")
    city_name = st.text_input("도시 이름 입력 (예: 인천, Busan)")
    if st.button("🔍 날씨 가져오기", key="weather_btn"):
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

# ──────────────────────────────────────────────────────────────────────────────
# 4) Chatbot (Vision)  ―  입력 메시지는 모두 화면에 유지
# ──────────────────────────────────────────────────────────────────────────────
enc = tiktoken.encoding_for_model("gpt-4o")
MAX_TOKENS        = 262_144
SUMMARY_THRESHOLD = 40
KEEP_RECENT       = 10

def num_tokens(messages: list) -> int:
    total = 0
    for m in messages:
        c = m["content"]
        if isinstance(c, list):
            for blk in c:
                total += len(enc.encode(
                    blk["text"] if blk["type"] == "text" else blk["image_url"]["url"]
                ))
        else:
            total += len(enc.encode(c))
    return total

@backoff.on_exception(backoff.expo, openai.RateLimitError, max_time=60, jitter=None)
def safe_chat_completion(messages, model="gpt-4o"):
    if num_tokens(messages) > MAX_TOKENS:
        raise ValueError("입력 토큰 초과")
    return openai.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=300,
        stream=True
    )

def compress_image(file, max_px=768, quality=85):
    img = Image.open(file)
    if max(img.size) > max_px:
        r = max_px / max(img.size)
        img = img.resize((int(img.width*r), int(img.height*r)), Image.LANCZOS)
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()

def summarize_history(history: list) -> str:
    prompt = [{"role":"system","content":"아래 대화를 3문장 이내로 요약해 주세요."}]
    prompt += history + [{"role":"user","content":"자, 이 대화 내용을 3문장 이내로 요약해 줘."}]
    res = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=prompt,
        max_tokens=200
    )
    return res.choices[0].message.content.strip()

def chatgpt_clone_section():
    st.subheader("💬 Chatbot (Vision)")

    # 세션 상태
    st.session_state.setdefault("chat_history",  [])  # UI 전용
    st.session_state.setdefault("model_history", [])  # GPT 호출용

    img_file = st.file_uploader("🖼️ 이미지 (선택)", type=["png", "jpg", "jpeg"])
    prompt   = st.chat_input("메시지를 입력하세요")

    # 1) model_history 요약(토큰 감축)
    if len(st.session_state.model_history) > SUMMARY_THRESHOLD:
        try:
            summary_txt = summarize_history(st.session_state.model_history)
            recent = st.session_state.model_history[-KEEP_RECENT:]
            st.session_state.model_history = [{"role":"assistant","content":summary_txt}] + recent
        except Exception as e:
            st.error(f"대화 요약 오류: {e}")

    # 2) 이전 대화 표시
    for msg in st.session_state.chat_history:
        role, content = msg["role"], msg["content"]
        with st.chat_message("user" if role=="user" else "assistant"):
            if isinstance(content, list):
                for blk in content:
                    if blk["type"] == "text":
                        st.write(blk["text"])
                    else:
                        st.image(blk["image_url"]["url"], caption="업로드 이미지")
            else:
                st.write(content)

    # 3) 입력 없으면 종료
    if img_file is None and not prompt:
        return

    # 4) 사용자 블록 구성
    user_blocks = []
    if prompt:
        user_blocks.append({"type":"text","text":prompt})
    if img_file:
        jpg_bytes = compress_image(img_file)
        st.image(jpg_bytes,
                 caption=f"미리보기 ({len(jpg_bytes)//1024} KB)",
                 use_container_width=True)
        b64 = base64.b64encode(jpg_bytes).decode()
        user_blocks.append({
            "type":"image_url",
            "image_url":{"url":f"data:image/jpeg;base64,{b64}"}
        })

    # 5) 히스토리 추가
    st.session_state.chat_history.append({"role":"user","content":user_blocks})
    st.session_state.model_history.append({"role":"user","content":user_blocks})

    # 6) GPT 호출
    try:
        resp = safe_chat_completion(st.session_state.model_history)
        buf = ""
        st.session_state.chat_history.append({"role":"assistant","content":""})
        st.session_state.model_history.append({"role":"assistant","content":""})
        with st.chat_message("assistant"):
            ph = st.empty()
            for chunk in resp:
                delta = chunk.choices[0].delta.content
                if delta:
                    buf += delta
                    ph.markdown(buf + "▌")
            ph.markdown(buf)
        st.session_state.chat_history[-1]["content"] = buf
        st.session_state.model_history[-1]["content"] = buf
    except openai.RateLimitError:
        st.error("⏳ 레이트 리밋, 잠시 후 시도해 주세요.")
    except Exception as e:
        st.error(f"OpenAI 호출 오류: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# 5) 댓글  (comments.csv)
# ──────────────────────────────────────────────────────────────────────────────
def comments_section():
    st.subheader("🗨️ 댓글 남기기")
    comments_file = "comments.csv"
    if not os.path.exists(comments_file):
        pd.DataFrame(columns=["timestamp","name","comment"])\
            .to_csv(comments_file, index=False, encoding="utf-8-sig")

    with st.form(key="comment_form", clear_on_submit=True):
        name    = st.text_input("이름", max_chars=50)



















