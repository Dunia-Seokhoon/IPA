# app.py  |  Streamlit 통합 데모
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import feedparser
import requests
from datetime import datetime, date
from urllib.parse import urlencode

# ──────────────── 여기서부터 API 키를 하드코딩 ────────────────
API_KEY = "GprdI3W07y8Ul7R0KwyRE0Beb1Y2wqtlBuvzWRqLqIZzEkR7xrPePc6CMQeD9FQAsTyQHh1V8NDK1md4ou4WGw=="

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
        st.write(df)
        nums = df.select_dtypes(include="number").columns.tolist()
        if nums:
            col = st.selectbox("Numeric 컬럼 선택", nums)
            fig, ax = plt.subplots()
            ax.hist(df[col], bins=10)
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
    else:
        st.info("CSV 파일을 올리면 데이터 미리보기와 히스토그램을 볼 수 있습니다.")

# ────────────────── 3) 동영상 업로드·재생 섹션 ──────────────────
def video_upload_section():
    st.subheader("📹 동영상 업로드 & 재생")
    video_file = st.file_uploader(
        "동영상 파일 업로드 (MP4 / MOV / AVI)",
        type=["mp4", "mov", "avi"],
        key="video_uploader",
    )
    if video_file:
        st.video(video_file)
    else:
        st.info("위 버튼으로 동영상 파일을 선택하세요.")

# ────────────────── 4) 선박 관제정보 조회 섹션 ──────────────────
def vessel_monitoring_section():
    st.subheader("🚢 해양수산부 선박 관제정보 조회")

    date_from = st.date_input("조회 시작일", value=date.today())
    date_to   = st.date_input("조회 종료일", value=date.today())
    page      = st.number_input("페이지 번호", min_value=1, value=1)
    per_page  = st.slider("한 번에 가져올 건수", 1, 1000, 100)

    if st.button("🔍 조회", key="vessel_btn"):
        BASE_URL = (
            "https://api.odcloud.kr/api/15128156/v1/"
            "uddi:fdcdb0d1-0296-4c3b-8087-8ab4bd4d5123"
        )
        params = {
            "serviceKey": API_KEY,
            "page":       page,
            "perPage":    per_page,
            "fromDate":   date_from.strftime("%Y-%m-%d"),
            "toDate":     date_to.strftime("%Y-%m-%d"),
        }

        with st.spinner("데이터를 불러오는 중..."):
            try:
                res = requests.get(BASE_URL, params=params)
                res.raise_for_status()
                data = res.json()
            except Exception as e:
                st.error(f"API 요청 실패: {e}")
                return

        if data.get("data"):
            df = pd.DataFrame(data["data"])
            total = data.get("totalCount", len(df))
            st.success(f"총 {total} 건 조회되었습니다.")
            st.dataframe(df)
        else:
            st.warning("조회된 데이터가 없습니다.")

        st.markdown(
            """
            **참고**  
            - API 명세: https://infuser.odcloud.kr/oas/docs?namespace=15128156/v1  
            """
        )

# ────────────────── 5) 오늘의 날씨 섹션 ──────────────────
def today_weather_section():
    st.subheader("☀️ 오늘의 날씨 조회")
    city = st.text_input("도시 이름을 입력하세요", value="Seoul")
    if st.button("🔍 날씨 가져오기", key="weather_btn"):
        coords = {
            "Seoul":   (37.5665, 126.9780),
            "Busan":   (35.1796, 129.0756),
            "Incheon": (37.4563, 126.7052),
        }
        lat, lon = coords.get(city, coords["Seoul"])
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&current_weather=true&timezone=Asia/Seoul"
        )
        with st.spinner(f"{city} 날씨 불러오는 중…"):
            resp = requests.get(url)
            resp.raise_for_status()
            w = resp.json()["current_weather"]

        st.markdown(f"### {city}의 현재 날씨")
        c1, c2, c3 = st.columns(3)
        c1.metric("🌡️ 기온(℃)", w["temperature"])
        c2.metric("💨 풍속(m/s)", w["windspeed"])
        c3.metric("🌫️ 풍향(°)", w["winddirection"])

# ────────────────── 6) 앱 레이아웃 (탭 구성) ──────────────────
st.set_page_config(page_title="통합 데모", layout="centered")
st.title("📈 통합 데모: 구글 뉴스 · 데이터 · 동영상 · 선박 · 날씨")

tab_news, tab_hist, tab_vid, tab_vessel, tab_weather = st.tabs(
    ["구글 뉴스", "데이터 히스토그램", "동영상 재생", "선박 관제정보", "오늘의 날씨"]
)

with tab_news:
    st.subheader("▶ 구글 뉴스 크롤링 (RSS)")
    kw  = st.text_input("검색 키워드", value="ESG", key="kw_input")
    num = st.slider("가져올 기사 개수", 5, 20, 10, key="num_slider")
    if st.button("최신 뉴스 보기", key="news_btn"):
        with st.spinner(f"‘{kw}’ 뉴스 불러오는 중…"):
            for item in fetch_google_news(kw, num):
                st.markdown(
                    f"- **[{item['source']} · {item['date']}]** "
                    f"[{item['title']}]({item['link']})"
                )

with tab_hist:
    sample_data_section()

with tab_vid:
    video_upload_section()

with tab_vessel:
    vessel_monitoring_section()

with tab_weather:
    today_weather_section()
