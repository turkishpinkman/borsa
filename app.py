import streamlit as st
import yfinance as yf
import pandas as pd
import google.generativeai as genai
import plotly.graph_objects as go

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Borsa Analiz Projesi", layout="wide")
st.title("📈 Borsa Veri Analiz Simülasyonu")

# --- SIDEBAR ---
st.sidebar.header("Kontrol Paneli")
symbol_input = st.sidebar.text_input("Hisse Kodu (Örn: GARAN.IS)", value="GARAN.IS")
analyze_button = st.sidebar.button("Verileri Getir")

# API Key
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
else:
    st.error("API Key Eksik! Streamlit Secrets ayarlarını yapın.")
    st.stop()

# --- 1. VERİ ÇEKME FONKSİYONU ---
@st.cache_data(ttl=300) # 5 dk önbellek
def get_data(symbol):
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="3mo") # Veriyi azalttık (Hız için)
        
        if hist.empty:
            return None, None, "Hisse bulunamadı. Sonuna .IS eklediniz mi?"

        # Teknik Hesaplamalar (Basitleştirilmiş)
        # RSI
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        hist['RSI'] = 100 - (100 / (1 + rs))
        
        # Hareketli Ortalama (50 Günlük)
        hist['SMA50'] = hist['Close'].rolling(window=50).mean()
        
        info = ticker.info
        
        # Haber Başlıkları
        news = ""
        if ticker.news:
            for n in ticker.news[:3]:
                news += f"- {n.get('title', '')}\n"
        
        return hist, info, news
    except Exception as e:
        return None, None, str(e)

# --- 2. AI YORUM FONKSİYONU (GÜVENLİ) ---
def get_ai_analysis(symbol, price, rsi, trend, news):
    # Prompt'u "Eğitim" kılıfına sokuyoruz
    prompt = f"""
    Rol yap: Sen bir üniversitede finans dersi veren bir profesörsün.
    Ben de senin öğrencinim. Aşağıdaki borsa verilerini kullanarak bana teknik analizin nasıl yorumlanacağını öğret.
    
    UYARI: Asla doğrudan "Al" veya "Sat" deme. Sadece verilerin ne anlama geldiğini anlat.
    Amaç tamamen eğitimdir.
    
    VERİLER:
    - Hisse: {symbol}
    - Fiyat: {price:.2f}
    - RSI: {rsi:.2f}
    - Trend Durumu: {trend}
    - Haberler: {news}
    
    AÇIKLAMA PLANIN:
    1. Teknik Göstergeler ne anlatıyor? (Aşırı alım/satım var mı?)
    2. Temel haberler fiyatı nasıl etkileyebilir?
    3. Teorik olarak bir yatırımcı bu tabloda nelere dikkat etmeli?
    """
    
    model = genai.GenerativeModel('gemini-3-flash-preview')
    
    # Tüm güvenlik filtrelerini kapatıyoruz
    safe = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
    
    try:
        response = model.generate_content(prompt, safety_settings=safe)
        
        # HATA YAKALAYICI: Cevap boş mu kontrol et
        if response.candidates and response.candidates[0].content.parts:
            return response.text
        else:
            return "⚠️ Yapay zeka bu hisse için yorum yapmaktan kaçındı (Finansal Filtre). Başka bir hisse deneyin."
            
    except Exception as e:
        return f"Bağlantı Hatası: {str(e)}"

# --- 3. ANA EKRAN ---
if analyze_button:
    with st.spinner('Veriler analiz ediliyor...'):
        hist, info, news = get_data(symbol_input)
        
    if hist is not None:
        last_price = hist['Close'].iloc[-1]
        last_rsi = hist['RSI'].iloc[-1]
        sma50 = hist['SMA50'].iloc[-1]
        
        # Trend Hesabı
        trend = "Yükseliş Trendi (Fiyat > 50 Günlük Ort)" if last_price > sma50 else "Düşüş Trendi (Fiyat < 50 Günlük Ort)"
        
        # Görselleştirme
        col1, col2 = st.columns(2)
        col1.metric("Son Fiyat", f"{last_price:.2f} TL")
        col2.metric("RSI (Güç)", f"{last_rsi:.2f}")
        
        st.write(f"**Sektör:** {info.get('sector', 'Belirsiz')}")
        
        # Grafik
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=hist.index,
                        open=hist['Open'], high=hist['High'],
                        low=hist['Low'], close=hist['Close'], name='Fiyat'))
        fig.update_layout(height=350, margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig, use_container_width=True)
        
        # AI Analizi Çağır
        st.subheader("🎓 Prof. AI Analizi")
        with st.spinner('Profesör notları hazırlıyor...'):
            comment = get_ai_analysis(symbol_input, last_price, last_rsi, trend, news)
            st.info(comment)
            
    else:
        st.error(f"Veri alınamadı: {news}")
