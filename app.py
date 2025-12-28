import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# 1. Nastavitve strani
st.set_page_config(page_title="Brand Sentiment Analysis 2023", layout="wide")

# 2. Nalaganje modela (Transformers)
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

sentiment_pipeline = load_sentiment_model()

def get_sentiment_data(text):
    if not isinstance(text, str) or text.strip() == "":
        return "Neznano", 0.0
    # Model DistilBERT
    result = sentiment_pipeline(text[:512])[0]
    sentiment = "Pozitivno" if result['label'] == 'POSITIVE' else "Negativno"
    confidence = result['score']
    return sentiment, confidence

# 3. Funkcija za dvojni Word Cloud
def prikazi_dvojni_wordcloud(df_mesec):
    pozitivna = df_mesec[df_mesec['Sentiment'] == 'Pozitivno']['review_text']
    negativna = df_mesec[df_mesec['Sentiment'] == 'Negativno']['review_text']
    
    stop_besede = {'in', 'je', 'da', 'pa', 'za', 'na', 'bi', 'se', 'iz', 'the', 'and', 'was', 'with', 'to', 'for', 'it', 'of', 'i', 'you', 'my', 'is', 'a'}
    
    col_wc1, col_wc2 = st.columns(2)
    with col_wc1:
        st.write("✅ **Pozitivno**")
        if not pozitivna.empty:
            wc_pos = WordCloud(width=400, height=300, background_color='white', colormap='Greens', stopwords=stop_besede).generate(" ".join(pozitivna.astype(str)))
            fig_pos, ax_pos = plt.subplots()
            ax_pos.imshow(wc_pos, interpolation='bilinear')
            ax_pos.axis("off")
            st.pyplot(fig_pos)
        else:
            st.write("Ni dovolj pozitivnih mnenj.")

    with col_wc2:
        st.write("❌ **Negativno**")
        if not negativna.empty:
            wc_neg = WordCloud(width=400, height=300, background_color='white', colormap='Reds', stopwords=stop_besede).generate(" ".join(negativna.astype(str)))
            fig_neg, ax_neg = plt.subplots()
            ax_neg.imshow(wc_neg, interpolation='bilinear')
            ax_neg.axis("off")
            st.pyplot(fig_neg)
        else:
            st.write("Ni dovolj negativnih mnenj.")

# 4. Nalaganje in sortiranje podatkov
@st.cache_data
def load_data():
    try:
        rev_df = pd.read_csv("reviews.csv")
        # Prisili v pravilen format in sortiraj, da bodo meseci v pravem vrstnem redu
        rev_df['date'] = pd.to_datetime(rev_df['date'])
        rev_df = rev_df.sort_values(by='date')
        
        test_df = pd.read_csv("testimonials.csv")
        prod_df = pd.read_csv("products.csv")
        return rev_df, test_df, prod_df
    except Exception:
        return None, None, None

reviews_df, testimonials_df, products_df = load_data()

# 5. Navigacija
st.sidebar.header("📍 Navigacija")
page = st.sidebar.radio("Izberi pogled:", ["Analiza sentimenta", "Izjave strank", "Katalog izdelkov"])

# Priprava za leto 2023
df_2023 = reviews_df[reviews_df['date'].dt.year == 2023].copy()
df_2023['Month'] = df_2023['date'].dt.month_name()
month_order = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

st.title("📈 Analiza ugleda blagovne znamke (2023)")

if page == "Analiza sentimenta":
    if not df_2023.empty:
        with st.spinner('AI analizira sentiment...'):
            results = df_2023['review_text'].apply(get_sentiment_data)
            df_2023['Sentiment'], df_2023['Confidence'] = zip(*results)

        # GRAFIČNI PRIKAZ LETA 2023
        st.header("📊 Analiza sentimenta ")
        c1, c2 = st.columns(2)
        with c1:
            fig_pie = px.pie(df_2023, names='Sentiment', color='Sentiment',
                            color_discrete_map={'Pozitivno':'#2ca02c', 'Negativno':'#d62728'},
                            title="Skupni sentiment")
            st.plotly_chart(fig_pie, use_container_width=True)
        with c2:
            fig_hist = px.histogram(df_2023, x='Month', color='Sentiment', barmode='group',
                                   category_orders={"Month": month_order},
                                   title="Število mnenj po mesecih")
            st.plotly_chart(fig_hist, use_container_width=True)

        st.divider()

        # MESECNI SLIDER
        st.header("🔍 Pregled sentimenta po mesecih")
        # Pokažemo samo mesece, ki imajo podatke (Jan, Feb, Mar, Apr, May)
        available_months = [m for m in month_order if m in df_2023['Month'].unique()]
        selected_month = st.select_slider("Izberi mesec za Word Cloud:", options=available_months)
        
        month_filtered = df_2023[df_2023['Month'] == selected_month].copy()

        # Word Cloudi
        prikazi_dvojni_wordcloud(month_filtered)
        
        # Tabela
        st.subheader(f"Seznam vseh ocen za {selected_month}")
        st.dataframe(month_filtered[['date', 'review_text', 'Sentiment', 'Confidence']], use_container_width=True)
    else:
        st.warning("V datoteki ni podatkov za leto 2023.")

elif page == "Izjave strank":
    st.header("💬 Izjave naših uporabnikov")
    for _, row in testimonials_df.iterrows():
        st.markdown(f"**{row['author']}** {'⭐' * int(row.get('rating', 5))}")
        st.write(f"*{row['text']}*")
        st.divider()

elif page == "Katalog izdelkov":
    st.header("📦 Katalog izdelkov")
    for _, row in products_df.iterrows():
        c1, c2 = st.columns([1, 3])
        with c1:
            if 'Slika_URL' in row and pd.notna(row['Slika_URL']):
                st.image(row['Slika_URL'], width=150)
            else:
                st.write("Ni slike")
        with c2:
            st.subheader(row['Ime izdelka'])
            st.write(f"**Cena:** {row['Cena']}")
            if 'Opis' in row:
                st.info(row['Opis'])
        st.divider()