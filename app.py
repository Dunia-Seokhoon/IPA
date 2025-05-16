import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup

# 1) 네이버 ESG 뉴스 크롤러
@st.cache_data(ttl=300)
def fetch_naver_esg_news():
    url = "https://search.naver.com/search.naver"
    params = {
        "where": "news",
        "query": "ESG",
        "sm": "tab_opt",
        "sort": 0,
        "start": 1
    }
    resp = requests.get(url, params=params, headers={
        "User-Agent": "Mozilla/5.0"
    })
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    items = []
    for li in soup.select("ul.list_news > li"):
        a = li.select_one("a.news_tit")
        if not a:
            continue
        title = a["title"]
        link  = a["href"]
        source_tag = li.select_one("a.info.press")
        source = source_tag.get_text(strip=True) if source_tag else ""
        date_tag = li.select_one("span.info")
        date = date_tag.get_text(strip=True) if date_tag else ""
        items.append({
            "title": title,
            "link": link,
            "source": source,
            "date": date
        })
    return items

# 2) 기본 샘플 CSV 히스토그램 UI
def sample_data_section():
    st.subheader("샘플 데이터 히스토그램")
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
        st.info("샘플 CSV 파일을 업로드하거나, 아래 버튼으로 뉴스만 확인하세요.")

# 3) Streamlit 앱 레이아웃
st.title("📈 ESG 뉴스 & 샘플 데이터 데모")

with st.expander("▶ ESG 뉴스 불러오기", expanded=True):
    if st.button("최신 ESG 뉴스 보기"):
        with st.spinner("뉴스를 불러오는 중..."):
            news = fetch_naver_esg_news()
        if news:
            for item in news:
                st.markdown(
                    f"- **[{item['source']} · {item['date']}]** "
                    f"[{item['title']}]({item['link']})"
                )
        else:
            st.warning("뉴스를 찾을 수 없습니다.")

st.markdown("---")
sample_data_section()

