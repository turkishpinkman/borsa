import streamlit as st
import yfinance as yf
import pandas as pd
import google.generativeai as genai
import plotly.graph_objects as go

# --- SAYFA AYARLARI ---
st.set_page_config(
    page_title="Borsa Asistanı v2.0",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 AI Borsa Asistanı (Lite Versiyon)")

# --- SIDEBAR ---
st.sidebar.header("Ayarlar")
symbol_input = st.sidebar.text_input("Hisse Kodu (Örn: THYAO.IS)", value="THYAO.IS")
analyze_button = st.sidebar.button("Analiz Et")

# API Key Kontrolü
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
else:
    st.error("API Key Eksik! Lütfen Secrets ayarlarını kontrol et.")
    st.stop()

# --- 1. FONKSİYON: VERİ ÇEKME (ÖNBELLEKLİ) ---
# Bu fonksiyon veriyi çeker ve önbelleğe alır. Böylece site donmaz.
@st.cache_data(ttl=3600) # 1 saat boyunca veriyi hatırla
def get_stock_data(symbol):
    try:
        ticker = yf.Ticker(symbol)
        # Veri periyodunu biraz kısalttık (daha az RAM kullanımı için)
        hist = ticker.history(period="6mo")
        
        if hist.empty:
            return None, None, "Veri bulunamadı."

        # Basit İndikatörler
        # RSI
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        hist['RSI'] = 100 - (100 / (1 + rs))
        
        # Ortalamalar
        hist['SMA50'] = hist['Close'].rolling(window=50).mean()
        
        # Haberler (Sadece başlıklar, daha hızlı olması için)
        news_list = ticker.news
        news_text = ""
        if news_list:
            for n in news_list[:3]:
                news_text += f"- {n.get('title', '')}\n"
        
        info = ticker.info
        return hist, info, news_text
        
    except Exception as e:
        return None, None, str(e)

# --- 2. FONKSİYON: AI ANALİZİ (ÖNBELLEKLİ) ---
@st.cache_data(show_spinner=False)
def get_ai_comment(symbol, price, rsi, trend, news):
    try:
        model = genai.GenerativeModel('gemini-3-flash-preview')
        
        prompt = f"""
        Analiz: {symbol}
        Fiyat: {price:.2f}
        RSI: {rsi:.2f}
        Trend: {trend}
        Haberler: {news}
        
        GÖREV:
        Yatırımcı için bu verileri kısaca yorumla.
        Teknik olarak alım mı satım mı bölgesinde?
        Riskler neler?
        Sonuç: Pozitif/Negatif/Nötr.
        (Cevabı kısa tut, maksimum 5 cümle.)
        """
        
        # Güvenlik ayarları (Bloklanmayı önlemek için)
        safe = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        
        response = model.generate_content(prompt, safety_settings=safe)
        return response.text
    except Exception as e:
        return f"AI Hatası: {str(e)}"

# --- ANA AKIŞ ---
if analyze_button:
    # 1. Adım: Veri Çekme
    with st.spinner('Veriler borsadan çekiliyor...'):
        hist, info, news = get_stock_data(symbol_input)
    
    if hist is not None:
        # Son Değerler
        last_price = hist['Close'].iloc[-1]
        last_rsi = hist['RSI'].iloc[-1]
        sma50 = hist['SMA50'].iloc[-1]
        trend = "Yükseliş" if last_price > sma50 else "Düşüş"
        
        # Üst Kartlar
        c1, c2, c3 = st.columns(3)
        c1.metric("Fiyat", f"{last_price:.2f}")
        c2.metric("RSI", f"{last_rsi:.2f}")
        c3.metric("Trend (50G)", trend)
        
        # 2. Adım: Grafik (Daha hafif ayarlar)
        st.subheader("Grafik")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=hist.index,
                        open=hist['Open'], high=hist['High'],
                        low=hist['Low'], close=hist['Close'], name='Fiyat'))
        fig.update_layout(height=350, margin=dict(l=0, r=0, t=0, b=0)) # Yüksekliği azalttık
        st.plotly_chart(fig, use_container_width=True)
        
        # 3. Adım: Yapay Zeka (En son çalışır)
        st.markdown("---")
        st.subheader("Yapay Zeka Yorumu")
        
        with st.spinner('Yapay zeka düşünüyor...'):
            comment = get_ai_comment(symbol_input, last_price, last_rsi, trend, news)
            st.success(comment)
            
    else:
        st.error(f"Veri hatası: {news}")
