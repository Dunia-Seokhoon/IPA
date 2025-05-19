# app.py  |  Streamlit 통합 데모
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import feedparser
import requests
from datetime import datetime, date
from urllib.parse import urlencode, quote_plus

# ──────────────── 하드코딩된 API 키 ────────────────
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

# ────────────────── 5) 오늘의 날씨 섹션 (동적 도시 검색) ──────────────────
def today_weather_section():
    st.subheader("☀️ 오늘의 날씨 조회")
    city = st.text_input("도시 이름을 입력하세요 (예: Seoul, 서울)", value="Seoul")
    if st.button("🔍 날씨 가져오기", key="weather_btn"):
        # 1) 지오코딩 API로 위경도 조회
        geo_url = (
            f"https://geocoding-api.open-meteo.com/v1/search?"
            f"name={quote_plus(city)}&count=1&language=ko&format=json"
        )
        with st.spinner("위경도 조회 중…"):
            geo_res = requests.get(geo_url)
            if geo_res.status_code != 200:
                st.error(f"지오코딩 API 오류: {geo_res.status_code}")
                st.text(geo_res.text)
                return
            geo_json = geo_res.json()

        results = geo_json.get("results")
        if not results:
            st.error("해당 도시를 찾을 수 없습니다.")
            return
        lat = results[0]["latitude"]
        lon = results[0]["longitude"]
        city_name = results[0].get("name_local", results[0]["name"])  # 현지명

        # 2) 날씨 API 호출
        weather_url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&current_weather=true"
            f"&hourly=relativehumidity_2m"
            f"&timezone=Asia/Seoul"
        )
        with st.spinner(f"{city_name} 날씨 불러오는 중…"):
            w_res = requests.get(weather_url)
            w_res.raise_for_status()
            w_js = w_res.json()

        cw = w_js.get("current_weather", {})
        temp     = cw.get("temperature")
        wind_spd = cw.get("windspeed")
        wind_dir = cw.get("winddirection")
        code     = cw.get("weathercode")

        # 날씨코드 텍스트 매핑
        wc_map = {
            0: "맑음", 1: "주로 맑음", 2: "부분적 구름", 3: "구름 많음",
            45: "안개", 48: "안개(입상)",
            51: "이슬비 약함", 53: "이슬비 보통", 55: "이슬비 강함",
            61: "빗방울 약함", 63: "빗방울 보통", 65: "빗방울 강함",
            80: "소나기 약함", 81: "소나기 보통", 82: "소나기 강함",
            95: "뇌우", 96: "약한 뇌우", 99: "강한 뇌우"
        }
        weather_desc = wc_map.get(code, "알 수 없음")

        # 습도 조회
        times = w_js["hourly"]["time"]
        hums  = w_js["hourly"]["relativehumidity_2m"]
        now_str = datetime.now().strftime("%Y-%m-%dT%H:00")
        humidity = None
        if now_str in times:
            humidity = hums[times.index(now_str)]

        # 화면에 출력
        st.markdown(f"### {city_name} 현재 날씨")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🌡️ 기온(℃)", temp)
        c2.metric("💨 풍속(m/s)", wind_spd)
        c3.metric("🌫️ 풍향(°)", wind_dir)
        c4.metric("💧 습도(%)", humidity if humidity is not None else "–")
        st.markdown(f"**날씨 상태:** {weather_desc}")

# ────────────────── 6) 앱 레이아웃 (탭 구성) ──────────────────
st.set_page_config(page_title="통합 데모", layout="centered")
st.title("📈 통합 데모: 뉴스 · 데이터 · 동영상 · 선박 · 날씨")

tabs = st.tabs([
    "구글 뉴스", "데이터 히스토그램", "동영상 재생",
    "선박 관제정보", "오늘의 날씨"
])

with tabs[0]:
    st.subheader("▶ 구글 뉴스 크롤링 (RSS)")
    kw  = st.text_input("검색 키워드", "ESG", key="kw_input")
    num = st.slider("가져올 기사 개수", 5, 20, 10, key="num_slider")
    if st.button("보기", key="news_btn"):
        with st.spinner(f"‘{kw}’ 뉴스 불러오는 중…"):
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

