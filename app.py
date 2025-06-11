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
    kw        = st.text_input("키워드를 입력하세요 | 검색 후 링크 복사 기능을 활용하세요 ! ", value="글로벌 ESG 현황")
    max_items = st.slider("가져올 기사 개수", 5, 100, 10)

    if st.button("보기", key="news_btn"):
        news = fetch_google_news(kw, max_items)
        if not news:
            st.info("검색 결과가 없습니다.")
            return

        # ① 결과 목록 (화면용)
        for it in news:
            st.markdown(
                f"- **[{it['source']}] · {it['date']}** "
                f"[{it['title']}]({it['link']})",
                unsafe_allow_html=True
            )

        # ② “번호. 제목 | 링크” 형식으로 문자열 생성
        links_str = "\n".join(
            f"{i+1}. {n['title']} | {n['link']}"
            for i, n in enumerate(news)
        )
        # 슬래시로 이어붙이고 싶다면:
        # links_str = links_str.replace("\n", " / ")

        # ③ 숨은 textarea + 복사 버튼
        components.html(
            f"""
            <textarea id="linksArea" style="opacity:0;position:absolute;left:-9999px;">
{links_str}
            </textarea>
            <button id="copyBtn"
                    style="margin-top:8px;padding:6px 12px;
                           background:#f44336;color:#fff;border:none;border-radius:4px;
                           cursor:pointer;font-weight:bold;">
                📋 {len(news)}개 링크 복사
            </button>
            <script>
            const btn  = document.getElementById("copyBtn");
            const area = document.getElementById("linksArea");
            btn.onclick = () => {{
                area.select();
                document.execCommand("copy");
                const old = btn.innerText;
                btn.innerText = "✅ 복사 완료!";
                setTimeout(()=>btn.innerText = old, 1500);
            }};
            </script>
            """,
            height=50,
        )

        # ④ 미리보기(선택)
        st.text_area("🔗 링크 미리보기", links_str, height=120)
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
    city_name = st.text_input("도시 이름 입력 (예: 인천,인천광역시, Busan)")
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


# ─── 4) Chatbot (gpt-4o) & 요약 기능 ─────────────────────────────────────────
enc = tiktoken.encoding_for_model("gpt-4o")
MAX_TOKENS        = 262_144        # gpt-4o 허용치
SUMMARY_THRESHOLD = 40             # 요약 트리거 턴 수
KEEP_RECENT       = 10             # 요약 후 남겨둘 최신 메시지 수

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
        r = max_px / max(img.size)
        img = img.resize((int(img.width*r), int(img.height*r)), Image.LANCZOS)
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()

def summarize_history(history: list) -> str:
    system_prompt = [{"role":"system","content":"아래 대화를 3문장 이내로 요약해 주세요."}]
    prompt = system_prompt + history + \
        [{"role":"user","content":"자, 이 대화 내용을 3문장 이내로 요약해 줘."}]
    res = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=prompt,
        max_tokens=200
    )
    return res.choices[0].message.content.strip()

def chatgpt_clone_section():
    st.subheader("💬 Chatbot (gpt-4o)")

    # ── 상태 초기화 ───────────────────────────────────────────────────────
    st.session_state.setdefault("chat_history",  [])  # UI 표시용(모두 저장)
    st.session_state.setdefault("model_history", [])  # 모델 호출용(요약 가능)

    img_file = st.file_uploader("🖼️ 이미지 (선택)", type=["png", "jpg", "jpeg"])
    prompt   = st.chat_input("메시지를 입력하세요")

    # ── 1) 필요 시 model_history 요약 ─────────────────────────────────────
    if len(st.session_state.model_history) > SUMMARY_THRESHOLD:
        try:
            summary_txt = summarize_history(st.session_state.model_history)
            recent      = st.session_state.model_history[-KEEP_RECENT:]
            st.session_state.model_history = \
                [{"role":"assistant","content":summary_txt}] + recent
        except Exception as e:
            st.error(f"대화 요약 중 오류가 발생했습니다: {e}")

    # ── 2) 이전 대화 화면 표시(chat_history 기준) ─────────────────────────
    for msg in st.session_state.chat_history:
        role, content = msg["role"], msg["content"]
        with st.chat_message("user" if role=="user" else "assistant"):
            if isinstance(content, list):               # 멀티블록(user)
                for blk in content:
                    if blk["type"] == "text":
                        st.write(blk["text"])
                    else:
                        st.image(blk["image_url"]["url"],
                                 caption="업로드 이미지")
            else:                                       # 단일 텍스트
                st.write(content)

    # ── 3) 입력 없으면 종료 ───────────────────────────────────────────────
    if img_file is None and not prompt:
        return

    # ── 4) 사용자 입력 블록 구성 ─────────────────────────────────────────
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

    # ── 5) 두 히스토리에 모두 추가 ───────────────────────────────────────
    st.session_state.chat_history.append({"role":"user", "content":user_blocks})
    st.session_state.model_history.append({"role":"user","content":user_blocks})

    # ── 6) 토큰 체크 & GPT 호출(model_history 사용) ──────────────────────
    if num_tokens(st.session_state.model_history) > MAX_TOKENS:
        st.error("⚠️ 토큰 한도를 초과했습니다. 오래된 대화를 삭제하거나 새 창을 시작해 주세요.")
        return

    try:
        resp = safe_chat_completion(st.session_state.model_history)
        buf  = ""
        # 미리 비어 있는 assistant 메시지 추가
        st.session_state.chat_history.append({"role":"assistant","content":""})
        st.session_state.model_history.append({"role":"assistant","content":""})

        with st.chat_message("assistant"):
            ph = st.empty()
            for chunk in resp:
                delta = chunk.choices[0].delta.content
                if delta:
                    buf += delta
                    ph.markdown(buf + "▌")   # 스트리밍 중
            ph.markdown(buf)

        # 두 히스토리에 최종 답변 반영
        st.session_state.chat_history[-1]["content"] = buf
        st.session_state.model_history[-1]["content"] = buf

    except openai.RateLimitError:
        st.error("⏳ 레이트 리밋에 걸렸습니다. 잠시 후 다시 시도해 주세요.")
    except Exception as e:
        st.error(f"OpenAI 호출 오류: {e}")


# ─── 5) 댓글 섹션 ─────────────────────────────────────────────────────────────
def comments_section():
    """
    로컬 CSV 파일(comments.csv)을 사용하여 댓글을 저장하고, 보여주는 섹션.
    """
    st.subheader("🗨️ 댓글 남기기")

    # 1) 댓글 파일 경로 설정
    comments_file = "comments.csv"

    # 2) 댓글을 저장할 CSV 파일이 없으면 헤더만 생성
    if not os.path.exists(comments_file):
        df_init = pd.DataFrame(columns=["timestamp", "name", "comment"])
        df_init.to_csv(comments_file, index=False, encoding="utf-8-sig")

    # 3) 댓글을 입력받을 UI (이름, 댓글 내용, 등록 버튼)
    with st.form(key="comment_form", clear_on_submit=True):
        name = st.text_input("이름", max_chars=50)
        comment = st.text_area("댓글 내용", height=100, max_chars=500)
        submitted = st.form_submit_button("등록")

    # 4) 사용자가 제출 버튼을 누르면 CSV에 저장
    if submitted:
        if not name.strip():
            st.warning("이름을 입력해 주세요.")
        elif not comment.strip():
            st.warning("댓글 내용을 입력해 주세요.")
        else:
            # 타임스탬프 생성
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # 새로운 댓글 DataFrame
            new_row = pd.DataFrame([{
                "timestamp": ts,
                "name": name.strip(),
                "comment": comment.strip()
            }])
            # CSV에 이어붙이기
            new_row.to_csv(comments_file, mode="a", header=False, index=False, encoding="utf-8-sig")
            st.success("댓글이 등록되었습니다!")

    # 5) 저장된 모든 댓글을 읽어서 화면에 표시
    try:
        all_comments = pd.read_csv(comments_file, encoding="utf-8-sig")
        # 최신순으로 표시하려면 아래처럼 정렬
        all_comments = all_comments.sort_values(by="timestamp", ascending=False)
        st.markdown("#### 전체 댓글")
        for _, row in all_comments.iterrows():
            st.markdown(f"- **[{row['timestamp']}] {row['name']}**: {row['comment']}")
    except Exception as e:
        st.error(f"댓글을 불러오는 중 오류가 발생했습니다: {e}")

#----6) ESG 활동 참여 기록 

def participation_section():
    st.subheader("🖊️ ESG 활동 참여")

    # ── 1) 항목 목록 ────────────────────────────────────────────────────────────
    BASE_ACTIVITIES = [
        "개인 텀블러·머그잔 사용",
        "종이 대신 디지털 문서 활용",
        "퇴근 전 멀티탭 ‘OFF’",
        "생활 속 이면지 사용",
        "분리배출 생활화 인증",
        "친환경 인증 제품·원두 선택",
        "점심시간 ‘잔반 제로’ 캠페인",
        "탄소배출 표시·친환경 배송 서비스 이용",
        "사내 일회용품 사용 줄이기",
    ]

    
    img_dir, csv_file = "participation_images", "participation.csv"
    os.makedirs(img_dir, exist_ok=True)

    # ── 2) CSV 헤더 순서 교정 ────────────────────────────────────────────────────
    expected_cols = ["timestamp","department","name","activity","image_filename"]
    if os.path.exists(csv_file):
        df0 = pd.read_csv(csv_file, nrows=0, encoding="utf-8-sig")
        if list(df0.columns) != expected_cols and set(df0.columns) == set(expected_cols):
            df_bad = pd.read_csv(csv_file, encoding="utf-8-sig")
            df_bad = df_bad[expected_cols]
            df_bad.to_csv(csv_file, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(columns=expected_cols).to_csv(csv_file, index=False, encoding="utf-8-sig")

    # ── 3) 활동 입력 방식 (폼 밖) ───────────────────────────────────────────────
    mode = st.radio(
        "활동 입력 방식",
        ["목록에서 선택", "직접 입력"],
        horizontal=True,
        key="reg_mode"
    )
    if mode == "목록에서 선택":
        activity = st.selectbox("기본 활동 항목 중 선택", BASE_ACTIVITIES, key="reg_select")
    else:
        activity = st.text_input("직접 입력: 활동 내용", placeholder="예) 사무실 LED 조명 교체", key="reg_text")

    st.markdown("---")

    # ── 4) 신규 등록 폼 ─────────────────────────────────────────────────────────
    with st.form(key="participation_form", clear_on_submit=True):
        dept      = st.text_input("참여 부서", max_chars=50)
        person    = st.text_input("성명", max_chars=30)
        up_img    = st.file_uploader("증명자료(이미지)", type=["png","jpg","jpeg"])
        submitted = st.form_submit_button("제출")

    if submitted:
        if not dept.strip():
            st.warning("참여 부서를 입력해 주세요.")
        elif not person.strip():
            st.warning("성명을 입력해 주세요.")
        elif not activity.strip():
            st.warning("활동 내용을 입력해 주세요.")
        elif up_img is None:
            st.warning("이미지를 업로드해 주세요.")
        else:
            ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
            ext       = os.path.splitext(up_img.name)[1].lower()
            safe_name = "".join(person.split())
            img_fname = f"{ts}_{safe_name}{ext}"
            # 이미지 저장
            with open(os.path.join(img_dir, img_fname), "wb") as f:
                f.write(up_img.getbuffer())

            # CSV에 올바른 순서로 기록
            pd.DataFrame([{
                "timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "department":    dept.strip(),
                "name":          person.strip(),
                "activity":      activity.strip(),
                "image_filename": img_fname
            }]).to_csv(csv_file, mode="a", header=False, index=False,
                       encoding="utf-8-sig")

            st.success("✅ 참여 정보가 등록되었습니다!")


    # ── 5) 저장된 데이터 로드 및 표시 ─────────────────────────────────────────
    try:
        all_data = pd.read_csv(csv_file, encoding="utf-8-sig")
        all_data = all_data.loc[:, expected_cols]  # 순서 보장
        all_data = all_data.sort_values(by="timestamp", ascending=False).reset_index(drop=True)

        # 이번주의 우수 ESG 사원 (가장 많이 등록한 이름)
        if not all_data.empty:
            top_name = all_data["name"].value_counts().idxmax()
            st.markdown(f"### 🏆 이번주의 우수 ESG 사원: **{top_name}**")

        # 다운로드 링크
        b64 = base64.b64encode(
            all_data.to_csv(index=False, encoding="utf-8-sig").encode()
        ).decode()
        st.markdown(
            f'<a href="data:file/csv;base64,{b64}" download="participation.csv">📥 CSV 다운로드</a>',
            unsafe_allow_html=True
        )

        st.dataframe(all_data, use_container_width=True)
        ...

        # ── 6) 데이터 수정(expander) ───────────────────────────────────────────
        with st.expander("✏️ 데이터 수정", expanded=False):
            if all_data.empty:
                st.info("수정할 데이터가 없습니다.")
            else:
                idx = st.selectbox(
                    "수정할 항목 선택",
                    all_data.index,
                    format_func=lambda i: f"{all_data.loc[i,'timestamp']} / {all_data.loc[i,'name']}"
                )
                if idx is not None:
                    cur = all_data.loc[idx]
                    # 수정용 활동 입력 방식
                    edit_mode = st.radio(
                        "활동 입력 방식",
                        ["목록에서 선택", "직접 입력"],
                        horizontal=True,
                        key=f"edit_mode_{idx}"
                    )
                    if edit_mode == "목록에서 선택":
                        new_act = st.selectbox(
                            "활동 항목",
                            BASE_ACTIVITIES,
                            index=BASE_ACTIVITIES.index(cur["activity"])
                                     if cur["activity"] in BASE_ACTIVITIES else 0,
                            key=f"edit_sel_{idx}"
                        )
                    else:
                        new_act = st.text_input(
                            "직접 입력: 활동 내용",
                            value=cur["activity"],
                            key=f"edit_text_{idx}"
                        )

                    new_dept = st.text_input("부서", value=cur["department"], key=f"edit_dept_{idx}")
                    new_name = st.text_input("성명", value=cur["name"], key=f"edit_name_{idx}")
                    new_img  = st.file_uploader("새 이미지 업로드(선택)", type=["png","jpg","jpeg"], key=f"edit_img_{idx}")

                    if st.button("저장", key=f"save_edit_{idx}"):
                        if not new_act.strip():
                            st.warning("활동명을 입력해 주세요.")
                        else:
                            img_fname = cur["image_filename"]
                            if new_img is not None:
                                # 구 이미지 삭제
                                old_p = os.path.join(img_dir, img_fname)
                                if os.path.exists(old_p):
                                    os.remove(old_p)
                                ext       = os.path.splitext(new_img.name)[1].lower()
                                img_fname = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{''.join(new_name.split())}{ext}"
                                with open(os.path.join(img_dir, img_fname), "wb") as f:
                                    f.write(new_img.getbuffer())

                            # 수정된 내용 덮어쓰기 (activity 컬럼도 제대로 변경)
                            all_data.loc[idx, ["department","name","activity","image_filename"]] = [
                                new_dept.strip(), new_name.strip(), new_act.strip(), img_fname
                            ]
                            all_data.to_csv(csv_file, index=False, encoding="utf-8-sig")
                            st.success("✅ 수정 완료")
                            st.experimental_rerun()

        # ── 7) 데이터 삭제(expander) ────────────────────────────────────────────
        with st.expander("🗑️ 데이터 삭제", expanded=False):
            if all_data.empty:
                st.info("삭제할 데이터가 없습니다.")
            else:
                del_idxs = st.multiselect(
                    "삭제할 항목 선택",
                    all_data.index,
                    format_func=lambda i: f"{all_data.loc[i,'timestamp']} / {all_data.loc[i,'name']}"
                )
                if st.button("삭제", key="delete_rows"):
                    for i in del_idxs:
                        p = os.path.join(img_dir, all_data.loc[i,"image_filename"])
                        if os.path.exists(p):
                            os.remove(p)
                    all_data = all_data.drop(del_idxs).reset_index(drop=True)
                    all_data.to_csv(csv_file, index=False, encoding="utf-8-sig")
                    st.success("🗑️ 삭제 완료")
                    st.experimental_rerun()

        # ── 8) 썸네일 + 정보 표시 ───────────────────────────────────────────────
        for _, row in all_data.iterrows():
            c1, c2 = st.columns([1, 4])
            with c1:
                img_path = os.path.join(img_dir, row["image_filename"])
                st.image(img_path if os.path.exists(img_path) else None,
                         width=80, caption=row["name"])
            with c2:
                st.write(
                    f"- **[{row['timestamp']}]** {row['department']} / {row['name']}  \n"
                    f"  🚩 _{row['activity']}_"
                )

    except Exception as e:
        st.error(f"참여 현황 오류: {e}")




# ─── 7) 영상 모음 섹션 ───────────────────────────────────────────────────────────
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

    # 3. 카페에서 ESG 실천하기 2탄
    st.markdown("#### 카페에서 ESG 실천하기 2탄")
    st.video("https://storage.googleapis.com/videoupload_icpa/%EC%B9%B4%ED%8E%98%EC%97%90%EC%84%9C%20%ED%9C%B4%EC%A7%80%20%EC%A0%81%EA%B2%8C%20%EC%82%AC%EC%9A%A9%ED%95%98%EA%B8%B0.mp4")

    # 4. 회의실에서 불 끄기 
    st.markdown("#### 회의실에서 불 끄기")
    st.video("https://storage.googleapis.com/videoupload_icpa/%ED%9A%8C%EC%9D%98%EC%8B%A4%EC%97%90%EC%84%9C%20%EB%B6%88%EB%81%84%EA%B8%B0.mp4")

    # 5.일회용품 사용 줄이기
    st.markdown("#### 일회용품 사용 줄이기")
    st.video("https://storage.googleapis.com/videoupload_icpa/%EC%9D%BC%ED%9A%8C%EC%9A%A9%ED%92%88%20%EC%82%AC%EC%9A%A9%20%EC%A4%84%EC%9D%B4%EA%B8%B0.mp4")
    
    # 5.분리수거장에서 ESG 실천하기
    st.markdown("#### 분리수거장에서 ESG 실천하기")
    st.video("https://storage.googleapis.com/videoupload_icpa/%EB%B6%84%EB%A6%AC%EC%88%98%EA%B1%B0%EC%9E%A5%EC%97%90%EC%84%9C%20%EC%8B%A4%EC%B2%9C%ED%95%98%EB%8A%94%20ESG%20.mp4")

    
# 1) 프로젝트 팀 소개 섹션 함수 추가
def project_team_intro_section():
    st.subheader("프로젝트 팀 소개")
    st.markdown("""
안녕하세요. 항만공사 일경험 프로젝트를 통해 ESG를 주제로 프로젝트를 진행하는 팀 INUS입니다.

저희 팀은 "ESG" 캠페인을 주제로 진행하고자 하던 중, 어떻게 하면 사내 구성원들의 참여를 유도할 수 있을까 고민하던 중에, 홈페이지를 간단하게 제작하여 캠페인을 진행하고자 하였습니다.

1주일간의 짧은 기간동안 캠페인을 진행하게 되었지만, 저희 팀의 멘토를 담당해주시는 임지영 대리님께서 도와주신 덕분에 저희의 노력이 빛을 보게 되었습니다. 인천 항만공사 구성원 분들의 많은 참여와, 해당 프로젝트를 진행함에 있어서 도와주신 분들께 고마움을 표하고자 합니다.

저희 팀이 제작한 사이트에 대해서 개략적으로 설명 드리자면, ESG 캠페인 활동을 기록할 수 있는 tab, 저희 팀에서 제작한 영상 컨텐츠를 감상할 수 있는 tab, 구글 뉴스 검색 및 링크를 복사할 수 있는 tab, 선박관제정보를 확인할 수 있는 tab, chat GPT를 기반으로 한 챗봇 tab, 여러 의견을 남길 수 있는 댓글 tab, 오늘의 날씨를 확인할 수 있는 tab으로 구성하였습니다.

팀 구성
- 멘토 : 임지영 대리 [인천항만공사]
- 팀장 : 박석훈 [인천대학교 동북아 물류대학원 융합물류시스템 전공]
- 팀원 : 이동민 [인천대학교 동북아 국제통상물류학부 전공]
- 팀원 : 김도현 [인천대학교 동북아 국제통상물류학부 전공]
- 팀원 : 김도윤 [인천대학교 경제학부 전공]
""")




# ─── 8) 앱 레이아웃 (탭 구성) ─────────────────────────────────────────────────────
st.set_page_config(page_title="인천항만공사 ESG 통합 포털", layout="centered")
st.title("📈 인천항만공사 ESG 통합 포털")

tabs = st.tabs([
    "ESG 활동 참여", "ESG 영상 모음","구글 뉴스", "선박 관제정보","챗봇[GPT-4o]", "댓글","오늘의 날씨", "프로젝트 팀 소개"
])

with tabs[0]:
    participation_section()

with tabs[1]:
    video_collection_section()

with tabs[2]:
    google_news_section()

with tabs[3]:
    vessel_monitoring_section()

with tabs[4]:
    chatgpt_clone_section()

with tabs[5]:
    comments_section()

with tabs[6]:
    today_weather_section()

with tabs[7]:
    project_team_intro_section()














