import streamlit as st
import pandas as pd
from services.preprocessing import preprocess_data
from services.predictor import predictor
import time
import nltk
from collections import Counter
import plotly.graph_objects as go
import plotly.express as px
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')

st.set_page_config(
    page_title="Fake vs Real News Detection",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed" 
)

# CSS Styling
st.markdown("""
<style>
    /* Global Font */
    html, body, [class*="css"] {
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Header Styles */
    .main-header {
        font-size: 3rem;
        color: #2c3e50;
        text-align: center;
        font-weight: 800;
        margin-bottom: 0px;
        animation: fadeIn 1.5s;
    }
    .sub-text {
        text-align: center;
        color: #7f8c8d;
        margin-bottom: 30px;
        font-size: 1.2rem;
    }
    
    /* Result Card Styles */
    .result-card {
        padding: 40px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        color: white;
        margin-bottom: 30px;
        transition: transform 0.3s ease;
    }
    .result-card:hover {
        transform: translateY(-5px);
    }
    .hoax-bg {
        background: linear-gradient(135deg, #FF416C 0%, #FF4B2B 100%);
    }
    .true-bg {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    
    /* Animations */
    @keyframes fadeIn {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }
    
    /* Input Area */
    .stTextArea textarea {
        font-size: 16px;
        border-radius: 12px;
        border: 2px solid #ecf0f1;
    }
    .stTextArea textarea:focus {
        border-color: #3498db;
        box-shadow: 0 0 10px rgba(52, 152, 219, 0.2);
    }
    
    /* Custom Button */
    .stButton button {
        border-radius: 50px;
        height: 55px;
        font-weight: bold;
        font-size: 18px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'text_input' not in st.session_state:
    st.session_state.text_input = ""


def reset_app():
    st.session_state.page = 'home'
    st.session_state.prediction_result = None
    st.session_state.text_input = ""

def process_prediction(text):
    data = {'text': [text]}
    df = pd.DataFrame(data)
    
    processed_input = preprocess_data(df)
    
    prediction = predictor.predict(processed_input)
    pred_label = prediction[0] # 0 = True, 1 = Hoax
    
    probs = predictor.predict_proba(processed_input)[0]
    confidence = probs[pred_label] * 100
    
    words = word_tokenize(text)
    char_count = len(text)
    word_count = len(words)
    
    stop_words = set(stopwords.words('english'))
    clean_tokens = [w.lower() for w in words if w.isalnum() and w.lower() not in stop_words]
    word_freq = Counter(clean_tokens).most_common(10)
    
    st.session_state.prediction_result = {
        'label': pred_label,
        'confidence': confidence,
        'char_count': char_count,
        'word_count': word_count,
        'word_freq': word_freq,
        'original_text': text
    }
    
    st.session_state.page = 'result'

def show_home():
    st.markdown('<div class="main-header">⚖️ Hoax News Detector</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-text">Paste your news article below to verify its credibility</div>', unsafe_allow_html=True)

    col_center = st.columns([1, 8, 1])
    with col_center[1]:
        text_val = st.text_area(
            "News Content", 
            height=300, 
            placeholder="Type or paste news article content here...",
            label_visibility="collapsed"
        )
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            if st.button("🔍 ANALYZE CONTENT", type="primary", use_container_width=True):
                if text_val.strip():
                    with st.spinner("Analyzing semantics and patterns..."):
                        time.sleep(1.2)
                        process_prediction(text_val)
                        st.rerun() # Refresh
                else:
                    st.warning("⚠️ Please enter some text first.")

    st.markdown("---")
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        st.info("ℹ️ **About**: This tool uses Machine Learning trained on thousands of verified articles.")
    with fc2:
        st.info("🛡️ **Privacy**: Your text is processed in real-time and not stored permanently.")
    with fc3:
        st.info("👨‍💻 **Developer**: Muhammad Abid Baihaqi Al Faridzi")

def show_result():
    res = st.session_state.prediction_result
    
    if st.button("← Check Another News"):
        reset_app()
        st.rerun()

    st.markdown('<div class="main-header">Analysis Report</div>', unsafe_allow_html=True)
    st.write("") # Spacer

    if res['label'] == 1:
        st.markdown(f"""
        <div class="result-card hoax-bg">
            <h1 style="margin:0; font-size: 4rem;">HOAX DETECTED 🚨</h1>
            <p style="font-size: 1.5rem; margin-top:10px; opacity:0.9;">System is <b>{res['confidence']:.1f}%</b> confident this is misinformation.</p>
        </div>
        """, unsafe_allow_html=True)
        st.toast("⚠️ Warning: Misinformation detected!", icon="🚨")
    else:
        st.markdown(f"""
        <div class="result-card true-bg">
            <h1 style="margin:0; font-size: 4rem;">LIKELY REAL ✅</h1>
            <p style="font-size: 1.5rem; margin-top:10px; opacity:0.9;">System is <b>{res['confidence']:.1f}%</b> confident this is credible.</p>
        </div>
        """, unsafe_allow_html=True)
        st.balloons()

    tab1, tab2, tab3 = st.tabs(["📊 Confidence & Stats", "🔠 Linguistic Analysis", "📝 Original Text"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Confidence Meter")
            gauge_color = "#FF4B2B" if res['label'] == 1 else "#38ef7d"
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = res['confidence'],
                number = {'suffix': "%", 'font': {'size': 50}},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': gauge_color},
                    'steps': [
                        {'range': [0, 50], 'color': "#ecf0f1"},
                        {'range': [50, 100], 'color': "#ffffff"}],
                }
            ))
            fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with c2:
            st.subheader("Text Statistics")
            st.write("") 
            col_stat1, col_stat2 = st.columns(2)
            col_stat1.metric("Word Count", res['word_count'], delta="Words")
            col_stat2.metric("Character Count", res['char_count'], delta="Chars")
            
    with tab2:
        st.subheader("Top Keywords Used")
        if res['word_freq']:
            df_freq = pd.DataFrame(res['word_freq'], columns=['Word', 'Frequency'])
            
            fig_bar = px.bar(
                df_freq, 
                x='Frequency', 
                y='Word', 
                orientation='h',
                color='Frequency',
                color_continuous_scale='Reds' if res['label'] == 1 else 'Teal',
                text='Frequency'
            )
            fig_bar.update_layout(yaxis=dict(autorange="reversed"), height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.warning("Text too short for keyword analysis.")

    with tab3:
        st.subheader("Analyzed Content")
        st.text_area("Read-only view:", value=res['original_text'], height=300, disabled=True)


if st.session_state.page == 'home':
    show_home()
elif st.session_state.page == 'result':
    show_result()