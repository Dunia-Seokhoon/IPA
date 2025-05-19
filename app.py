import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import feedparser
import requests
from datetime import datetime, date
from urllib.parse import urlencode
from urllib.parse import urlencode, quote_plus  # ← quote_plus 추가

API_KEY = os.getenv("ODCLOUD_API_KEY", "GprdI3W07y8Ul7R0KwyRE0Beb1Y2wqtlBuvzWRqLqIZzEkR7xrPePc6CMQeD9FQAsTyQHh1V8NDK1md4ou4WGw==")

# Hugging Face Inference API 설정
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "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")
HF_API_URL   = os.getenv("HF_API_URL", "https:////huggingface.co/kakaocorp/kanana-nano-2.1b-instruct")

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
            "title": entry.title,
            "link":  entry.link,
            "source": source,
            "date":   pub_date,
        })
    return items

# ────────────────── 2) CSV 히스토그램 섹션 ──────────────────
def sample_data_section():
    st.subheader("📊 샘플 데이터 히스토그램")
    uploaded_file = st.file_uploader("CSV 파일 업로드 (optional)", type=["csv"], key="hist_csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df)
        nums = df.select_dtypes(include="number").columns.tolist()
        if nums:
            col = st.selectbox("Numeric 컬럼 선택", nums, key="hist_col")
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
    video_file = st.file_uploader("동영상 파일 업로드 (MP4/MOV/AVI)", type=["mp4","mov","avi"], key="vid_file")
    if video_file:
        st.video(video_file)
    else:
        st.info("파일을 선택해 주세요.")

# ────────────────── 4) 선박 관제정보 조회 섹션 ──────────────────
def vessel_monitoring_section():
    st.subheader("🚢 해양수산부 선박 관제정보 조회")
    date_from = st.date_input("조회 시작일", date.today(), key="vessel_from")
    date_to   = st.date_input("조회 종료일", date.today(), key="vessel_to")
    page      = st.number_input("페이지 번호", 1, 1000, 1, key="vessel_page")
    per_page  = st.slider("한 번에 가져올 건수", 1, 1000, 100, key="vessel_perpage")
    if st.button("🔍 조회", key="vessel_search"):
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
                st.text(res.text)
                return
            data = res.json()
        if data.get("data"):
            df = pd.DataFrame(data["data"])
            st.success(f"총 {data.get('totalCount', len(df))} 건 조회되었습니다.")
            st.dataframe(df)
        else:
            st.warning("조회된 데이터가 없습니다.")

def today_weather_section():
    st.subheader("☀️ 오늘의 날씨 조회")
    city_name = st.text_input("도시 이름 입력 (예: 서울, Seoul, 부산, Busan)", key="weather_city_input")
    if st.button("🔍 날씨 가져오기", key="weather_search"):
        if not city_name:
            st.warning("도시 이름을 입력해 주세요.")
            return

        # 한글·영어 모두 지원하도록 URL-encode
        q_name = quote_plus(city_name)

        # 한국어 결과 우선으로 language=ko 추가
        geo_url = (
            f"https://geocoding-api.open-meteo.com/v1/search?"
            f"name={q_name}&count=5&language=ko"
        )
        with st.spinner("위치 정보 검색 중…"):
            geo_res = requests.get(geo_url)
        if geo_res.status_code != 200:
            st.error("지오코딩 API 호출 실패")
            return

        geo_data = geo_res.json().get("results")
        if not geo_data:
            st.warning("해당 도시를 찾을 수 없습니다.")
            return

        loc = geo_data[0]
        lat  = loc["latitude"]
        lon  = loc["longitude"]
        display_name = f"{loc.get('name')}, {loc.get('country')}"

        # 날씨 정보 조회
        weather_url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&current_weather=true"
            f"&hourly=relativehumidity_2m"
            f"&timezone=auto"
        )
        with st.spinner(f"{display_name} 날씨 불러오는 중…"):
            w_res = requests.get(weather_url)
        if w_res.status_code != 200:
            st.error("날씨 API 호출 실패")
            return

        js = w_res.json()
        cw = js.get("current_weather", {})
        temp, wind_spd, wind_dir, code = (
            cw.get("temperature"),
            cw.get("windspeed"),
            cw.get("winddirection"),
            cw.get("weathercode")
        )

        # 날씨 코드 매핑
        wc_map = {
            0:"맑음",1:"주로 맑음",2:"부분적 구름",3:"구름 많음",
            45:"안개",48:"안개(입상)",
            51:"이슬비 약함",53:"이슬비 보통",55:"이슬비 강함",
            61:"빗방울 약함",63:"빗방울 보통",65:"빗방울 강함",
            80:"소나기 약함",81:"소나기 보통",82:"소나기 강함",
            95:"뇌우",96:"약한 뇌우",99:"강한 뇌우"
        }
        desc = wc_map.get(code, "알 수 없음")

        # 습도 얻기
        times = js.get("hourly", {}).get("time", [])
        hums  = js.get("hourly", {}).get("relativehumidity_2m", [])
        now   = datetime.now().strftime("%Y-%m-%dT%H:00")
        humidity = hums[times.index(now)] if now in times else None

        # 결과 출력
        st.markdown(f"### {display_name} 현재 날씨")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🌡️ 기온(℃)", temp)
        c2.metric("💨 풍속(m/s)", wind_spd)
        c3.metric("🌫️ 풍향(°)", wind_dir)
        c4.metric("💧 습도(%)", humidity if humidity is not None else "–")
        st.markdown(f"**날씨 상태:** {desc}")
# ────────────────── 6) LLM 테스트 (Hugging Face Inference API) ──────────────────) LLM 테스트 (Hugging Face) ──────────────────
@st.cache_resource
def generate_with_hf(prompt: str) -> str:
    if not HF_API_TOKEN:
        return "⚠️ HF_API_TOKEN 환경변수가 설정되지 않았습니다."
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}", "Content-Type": "application/json"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 256}}
    try:
        r = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"]
        if isinstance(data, list) and data and "generated_text" in data[0]:
            return data[0]["generated_text"]
        return str(data)
    except Exception as e:
        return f"HF 호출 오류: {e}"

def llm_section():
    st.subheader("🤖 LLM 테스트 (Hugging Face Inference API)")
    prompt = st.text_area("프롬프트 입력", height=150, key="llm_prompt")
    if st.button("생성", key="hf_generate"):
        with st.spinner("API 호출 중…"):
            out = generate_with_hf(prompt)
        st.markdown("### 응답")
        st.write(out)
    st.info("⚙️ 사용 전 HF_API_TOKEN과 HF_API_URL을 Secrets에 설정해주세요.")

# ────────────────── 7) 앱 레이아웃 (탭 구성) ──────────────────
st.set_page_config(page_title="통합 데모", layout="centered")
st.title("📈 통합 데모: 뉴스·데이터·동영상·선박·날씨·LLM")

tabs = st.tabs(["구글 뉴스", "데이터 히스토그램", "동영상 재생", "선박 관제정보", "오늘의 날씨", "LLM 테스트"])
with tabs[0]:
    st.subheader("▶ 구글 뉴스 크롤링 (RSS)")
    kw = st.text_input("검색 키워드", "ESG", key="news_kw")
    num = st.slider("가져올 기사 개수", 5, 20, 10, key="news_num")
    if st.button("보기", key="news_btn"):
        with st.spinner(f"‘{kw}’ 뉴스 로딩…"):
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




