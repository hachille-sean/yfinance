import os
import sys
import locale
import streamlit as st

# ==========================================
# 🚨 0. 파이썬 전역 인코딩 강제 덮어쓰기 (Monkey Patch)
# 이 부분이 LangChain 내부의 인코딩 에러를 원천 차단합니다.
# ==========================================
def getpreferredencoding(do_setlocale=True):
    return "utf-8"

locale.getpreferredencoding = getpreferredencoding
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except AttributeError:
    pass

# --- 이 아래부터 나머지 패키지들을 임포트합니다 ---
import yfinance as yf
#from ddgs import DDGS
from duckduckgo_search import DDGS
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

# ==========================================
# ⚙️ 1. 페이지 및 API 설정
# ==========================================
st.set_page_config(page_title="Sean's 주식 애널리스트", page_icon="📈", layout="centered")

# 수정된 부분: secrets.toml에서 API 키를 읽어와 환경 변수에 설정합니다.
# 이렇게 설정하면 LangChain의 ChatGoogleGenerativeAI가 자동으로 이 환경 변수를 인식합니다.
if "GEMINI_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]
else:
    st.error("API 키를 찾을 수 없습니다. secrets.toml 파일이나 Streamlit Cloud의 Secrets 설정을 확인해 주세요.")
    st.stop() # 키가 없으면 실행 중지

# ==========================================
# 🛠️ 2. 도구(Tools) 정의
# ==========================================
@tool
def get_stock_price(ticker: str) -> str:
    """주어진 티커(종목 코드)의 최근 5일간 주식 가격 및 전일 대비 변동폭을 가져옵니다."""
    try:
        stock = yf.Ticker(ticker)
        historical_data = stock.history(period='6d')
        
        if historical_data.empty or len(historical_data) < 2:
            return f"{ticker}의 가격 정보를 찾을 수 없습니다. 올바른 티커인지 확인해주세요."
        
        historical_data['Change'] = historical_data['Close'].diff()
        historical_data['Pct_Change'] = historical_data['Close'].pct_change() * 100
        last_5_days = historical_data.tail(5)
        
        result_text = f"[{ticker} 최근 5거래일 종가]\n"
        for date, row in last_5_days.iterrows():
            date_str = date.strftime('%Y-%m-%d')
            price = row['Close']
            change = row['Change']
            pct_change = row['Pct_Change']
            
            if change > 0:
                result_text += f"- {date_str}: {price:,.2f} (+{change:,.0f}, +{pct_change:.1f}%)\n"
            elif change < 0:
                result_text += f"- {date_str}: {price:,.2f} ({change:,.0f}, {pct_change:.1f}%)\n"
            else:
                result_text += f"- {date_str}: {price:,.2f} (0, 0.0%)\n"
                
        return result_text
    except Exception as e:
        return f"가격을 가져오는 중 오류가 발생했습니다: {e}"

@tool
def search_news(query: str) -> str:
    """주식 종목이나 회사 이름에 대한 최신 인터넷 뉴스를 검색합니다."""
    try:
        results = DDGS().text(f"{query} 주식 경제 뉴스", max_results=3)
        if not results:
            return "관련 뉴스를 찾을 수 없습니다."
        
        news_text = "[최신 뉴스 검색 결과]\n"
        for i, res in enumerate(results):
            # 딕셔너리 구조 변경에 대비해 .get() 사용
            title = res.get('title', '제목 없음')
            body = res.get('body', '내용 없음')
            news_text += f"{i+1}. 제목: {title}\n   내용 요약: {body}\n"
            
        return news_text
    except Exception as e:
        return f"뉴스 검색 중 오류 발생: {e}"

# ==========================================
# 🧠 3. AI 에이전트 초기화 (캐시 적용)
# ==========================================
@st.cache_resource
def load_agent():
    # 최신 모델 권장: gemini-2.5-flash 또는 gemini-2.5-pro
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    tools = [get_stock_price, search_news]
    # LangGraph를 사용하여 에이전트 생성
    return create_react_agent(llm, tools)

agent = load_agent()

system_prompt = (
    "당신은 월스트리트의 날카로운 통찰력을 가진 수석 AI 주식 애널리스트입니다. "
    "1. [get_stock_price] 도구로 최근 주가 흐름을 파악합니다. (한국 주식은 '.KS' 또는 '.KQ' 사용)\n"
    "2. [search_news] 도구로 해당 기업의 최신 이슈를 검색합니다.\n"
    "3. [중요] 답변을 작성할 때, 반드시 맨 처음에 도구가 반환한 '최근 5거래일 종가' 데이터를 있는 그대로 표기하세요.\n"
    "4. 그 다음, 가격 데이터와 뉴스 데이터를 종합하여 주가 변동 원인과 향후 전망을 '전문가 리포트' 형식으로 작성해 주세요."
)

# ==========================================
# 🌐 4. 웹 UI (화면 그리기)
# ==========================================
st.title("📈 Sean's AI 주식 애널리스트")
st.markdown("관심 있는 주식 종목을 입력하면, AI가 실시간 가격과 뉴스를 분석해 리포트를 작성합니다.")

# 대화 기록 저장
if "messages" not in st.session_state:
    st.session_state.messages = []

# 이전 대화 내용 출력
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==========================================
# 💬 5. 채팅 입력 및 AI 응답 처리
# ==========================================
if user_input := st.chat_input("종목을 입력하세요 (예: 테슬라 주가 어때?, 카카오 035720.KS 분석해 줘)"):
    
    # 사용자 입력 출력 및 저장
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # AI 응답 처리
    with st.chat_message("assistant"):
        with st.spinner("AI가 주가 차트와 최신 뉴스를 융합하여 분석 중입니다..."):
            
            # 매 요청마다 시스템 프롬프트를 주입하여 역할 고정
            inputs = {
                "messages": [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_input)
                ]
            }
            
            try:
                # 에이전트 실행
                response = agent.invoke(inputs)
                final_content = response["messages"][-1].content
                
                # 결과 텍스트 추출 (LangGraph 응답 구조 처리)
                report_text = ""
                if isinstance(final_content, list):
                    for block in final_content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            report_text += block["text"]
                        elif isinstance(block, str):
                            report_text += block
                else:
                    report_text = final_content
                
                # 화면에 리포트 출력 및 저장
                st.markdown(report_text)
                st.session_state.messages.append({"role": "assistant", "content": report_text})
                
            except Exception as e:
                error_msg = f"❌ 분석 중 오류가 발생했습니다: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})