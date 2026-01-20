import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import google.generativeai as genai

# --- 1. AYARLAR (EN ÜSTTE OLMALI) ---
st.set_page_config(
    page_title="Finansal Analiz Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed" # Mobilde yer kaplamasın diye kapalı başlar
)

# --- 2. API KONTROL ---
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
else:
    st.error("API Anahtarı eksik.")
    st.stop()

# --- 3. VERİ MOTORU ---
@st.cache_data(ttl=120) # 2 dk önbellek
def get_clean_data(symbol):
    try:
        ticker = yf.Ticker(symbol)
        # Veriyi çek
        hist = ticker.history(period="6mo")
        if hist.empty: return None

        # Hesaplamalar
        # RSI
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        hist['RSI'] = 100 - (100 / (1 + rs))
        
        # Ortalamalar
        hist['SMA50'] = hist['Close'].rolling(window=50).mean()
        hist['SMA200'] = hist['Close'].rolling(window=200).mean()
        
        # Son Veri Noktası
        curr = hist.iloc[-1]
        prev = hist.iloc[-2]
        
        # Değişim
        change_val = curr['Close'] - prev['Close']
        change_pct = (change_val / prev['Close']) * 100
        
        return {
            "df": hist,
            "price": curr['Close'],
            "change_val": change_val,
            "change_pct": change_pct,
            "rsi": curr['RSI'],
            "sma50": curr['SMA50'],
            "sma200": curr['SMA200'],
            "name": ticker.info.get('shortName', symbol)
        }
    except:
        return None

def get_market_comment(data):
    # İsimsiz analiz (Filtreye takılmamak için)
    trend = "Yükseliş" if data['price'] > data['sma200'] else "Düşüş"
    
    prompt = f"""
    Sen kıdemli bir portföy yöneticisisin. Aşağıdaki teknik verileri yorumla.
    Asla sohbet etme, direkt sadede gel.
    
    VERİLER:
    - Fiyat: {data['price']:.2f}
    - Trend (200G Ort): {trend}
    - RSI: {data['rsi']:.2f} (30 altı ucuz, 70 üstü pahalı bölge)
    - 50 Günlük Ort: {data['sma50']:.2f}
    
    İSTENEN FORMAT (Markdown):
    **Teknik Görünüm:** (Tek cümlede durum)
    **Risk Analizi:** (RSI ve ortalamalara göre risk durumu)
    **Stratejik Yorum:** (Yatırımcı neye dikkat etmeli? Destek/Direnç mantığı)
    """
    
    model = genai.GenerativeModel('gemini-3-flash-preview')
    safe = [{"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}]
    try:
        response = model.generate_content(prompt, safety_settings=safe)
        return response.text
    except:
        return "Bağlantı sorunu nedeniyle yorum yapılamadı."

# --- 4. ARAYÜZ (NATIVE STREAMLIT) ---

# Üst Başlık
st.title("Piyasa Analiz Paneli")

# Input Alanı (Ana ekranda üstte dursun, mobilde kolay erişim)
col_input, col_btn = st.columns([3, 1])
with col_input:
    symbol = st.text_input("Hisse Kodu", value="THYAO.IS", label_visibility="collapsed", placeholder="Hisse Kodu (Örn: GARAN.IS)")
with col_btn:
    btn = st.button("Analiz Et", type="primary", use_container_width=True)

if btn:
    data = get_clean_data(symbol)
    
    if data:
        # --- A. ÖZET BİLGİLER (KPI) ---
        # Mobilde 2 satır, masaüstünde 4 sütun
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        
        kpi1.metric("Fiyat", f"{data['price']:.2f} ₺", f"{data['change_pct']:.2f}%")
        kpi2.metric("RSI (14)", f"{data['rsi']:.2f}", "Güç Endeksi")
        kpi3.metric("50 G. Ort", f"{data['sma50']:.2f} ₺")
        kpi4.metric("200 G. Ort", f"{data['sma200']:.2f} ₺")
        
        st.markdown("---")
        
        # --- B. GRAFİK (TAM EKRAN) ---
        # Plotly'nin kendi native teması mobilde en iyisidir.
        fig = go.Figure()
        
        # Mumlar
        fig.add_trace(go.Candlestick(
            x=data['df'].index,
            open=data['df']['Open'], high=data['df']['High'],
            low=data['df']['Low'], close=data['df']['Close'],
            name='Fiyat'
        ))
        
        # Ortalamalar (Sadece çizgiler)
        fig.add_trace(go.Scatter(x=data['df'].index, y=data['df']['SMA50'], line=dict(color='orange', width=1), name='50 G. Ort'))
        fig.add_trace(go.Scatter(x=data['df'].index, y=data['df']['SMA200'], line=dict(color='blue', width=1), name='200 G. Ort'))
        
        fig.update_layout(
            height=450, # Mobilde ideal yükseklik
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_rangeslider_visible=False, # Alttaki slider mobilde yer kaplar, kapattık
            legend=dict(orientation="h", y=1, x=0)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # --- C. YAPAY ZEKA RAPORU ---
        # Expander içine alıyoruz, böylece ekranı hemen kaplamaz, isteyen tıklar okur.
        with st.status("Yapay Zeka Raporu Hazırlanıyor...", expanded=True) as status:
            comment = get_market_comment(data)
            st.markdown(comment)
            status.update(label="Analiz Tamamlandı", state="complete", expanded=True)
            
    else:
        st.error("Veri bulunamadı. Lütfen kodu kontrol edin (BIST için .IS ekleyin).")
