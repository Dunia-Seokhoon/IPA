# app.py  |  Streamlit 통합 데모
# ─────────────────────────────────────────────

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import feedparser
from bs4 import BeautifulSoup
from datetime import datetime
from urllib.parse import quote_plus  # URL 인코딩용

# ────────────────── 1) 뉴스 크롤러 (Google News RSS)
@st.cache_data(ttl=300)
def fetch_google_news(keyword: str, max_items: int = 10):
    """
    Google News RSS 피드에서 keyword 검색 결과를 최대 max_items개 가져온다.
    반환: [{title, link, source, date}, …]
    """
    # 띄어쓰기가 있는 검색어도 안전하게 처리
    encoded = quote_plus(keyword)
    rss_url = (
        "https://news.google.com/rss/search?"
        f"q={encoded}&hl=ko&gl=KR&ceid=KR:ko"
    )
    feed = feedparser.parse(rss_url)

    items = []
    for entry in feed.entries[:max_items]:
        pub_date = datetime(*entry.published_parsed[:6]).strftime("%Y-%m-%d")
        source = entry.source.title if "source" in entry else ""
        items.append({
            "title": entry.title,
            "link" : entry.link,
            "source": source,
            "date"  : pub_date,
        })
    return items

# ────────────────── 2) CSV 히스토그램 섹션
def sample_data_section():
    st.subheader("📊 샘플 데이터 히스토그램")
    uploaded_file = st.file_uploader("CSV 파일 업로드 (optional)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write(df)

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            col = st.selectbox("Numeric 컬럼 선택", numeric_cols)
            fig, ax = plt.subplots()
            ax.hist(df[col], bins=10)
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
    else:
        st.info("CSV 파일을 올리면 데이터 미리보기와 히스토그램을 볼 수 있습니다.")

# ────────────────── 3) 동영상 업로드·재생 섹션
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

# ────────────────── 4) 앱 레이아웃 (탭 구성)
st.set_page_config(page_title="통합 데모", layout="centered")
st.title("📈 통합 데모: 구글 뉴스 · 데이터 · 동영상")

tab_news, tab_hist, tab_vid = st.tabs(
    ["구글 뉴스", "데이터 히스토그램", "동영상 재생"]
)

with tab_news:
    st.subheader("▶ 구글 뉴스 크롤링 (RSS)")
    keyword = st.text_input("검색 키워드", value="ESG", key="kw_input")
    num     = st.slider("가져올 기사 개수", 5, 20, 10, key="num_slider")

    if st.button("최신 뉴스 보기", key="news_btn"):
        with st.spinner(f"‘{keyword}’ 뉴스 불러오는 중…"):
            news_items = fetch_google_news(keyword, num)

        if news_items:
            for item in news_items:
                st.markdown(
                    f"- **[{item['source']} · {item['date']}]** "
                    f"[{item['title']}]({item['link']})"
                )
        else:
            st.warning("뉴스를 찾을 수 없습니다.")

with tab_hist:
    sample_data_section()

with tab_vid:
    video_upload_section()

