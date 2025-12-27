import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline

# Nastavitve strani
st.set_page_config(page_title="Brand Sentiment Analysis 2023", layout="wide")

# Naslov aplikacije
st.title("📈 Analiza ugleda blagovne znamke (2023)")
st.markdown("Aplikacija izvaja analizo sentimenta na podlagi ocen strank z uporabo globokega učenja (BERT/DistilBERT).")

# 1. Nalaganje modela za globoko učenje (Transformers)
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

sentiment_pipeline = load_sentiment_model()

# 2. Posodobljena funkcija za analizo sentimenta s Confidence Score
def get_sentiment_data(text):
    if not isinstance(text, str) or text.strip() == "":
        return "Neznano", 0.0
    # Transformers vrnejo labelo in score (verjetnost/zaupanje)
    result = sentiment_pipeline(text[:512])[0]
    
    sentiment = "Pozitivno" if result['label'] == 'POSITIVE' else "Negativno"
    confidence = result['score']
    
    return sentiment, confidence

# 3. Nalaganje podatkov iz CSV datotek
@st.cache_data
def load_data():
    try:
        reviews_df = pd.read_csv("reviews.csv")
        reviews_df['date'] = pd.to_datetime(reviews_df['date'])
        testimonials_df = pd.read_csv("testimonials.csv")
        products_df = pd.read_csv("products.csv")
        return reviews_df, testimonials_df, products_df
    except Exception as e:
        return None, None, None

reviews_df, testimonials_df, products_df = load_data()

if reviews_df is None:
    st.error("Napaka pri branju CSV datotek. Prepričaj se, da si najprej zagnal scrape_data.py!")
    st.stop()

# 4. Navigacija na stranski vrstici
st.sidebar.header("Nadzorna plošča")
page = st.sidebar.radio("Izberi pogled:", ["Analiza sentimenta (Ocene)", "Izjave strank", "Katalog izdelkov"])

# Filtriranje za leto 2023
df_2023 = reviews_df[reviews_df['date'].dt.year == 2023].copy()

if page == "Analiza sentimenta (Ocene)":
    st.header("⭐ Analiza ocen za leto 2023")
    
    if not df_2023.empty:
        with st.spinner('Model analizira sentiment in stopnjo zaupanja...'):
            # Pridobivanje obeh vrednosti hkrati
            results = df_2023['review_text'].apply(get_sentiment_data)
            df_2023['Sentiment'], df_2023['Confidence'] = zip(*results)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Porazdelitev mnenj")
            fig_pie = px.pie(df_2023, names='Sentiment', color='Sentiment',
                            color_discrete_map={'Pozitivno':'#2ca02c', 'Negativno':'#d62728'})
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            st.subheader("Število mnenj po mesecih")
            df_2023['Month'] = df_2023['date'].dt.month_name()
            month_order = ["January", "February", "March", "April", "May", "June", 
                           "July", "August", "September", "October", "November", "December"]
            fig_bar = px.histogram(df_2023, x='Month', color='Sentiment', barmode='group',
                                  category_orders={"Month": month_order})
            st.plotly_chart(fig_bar, use_container_width=True)

        # NAPREDNO: Filtriranje po mesecu z grafom Positive vs. Negative in Confidence Tooltipom
        st.divider()
        available_months = [m for m in month_order if m in df_2023['Month'].unique()]
        selected_month = st.selectbox("Izberi mesec za podroben grafikon sentimenta:", options=available_months)
        
        month_filtered = df_2023[df_2023['Month'] == selected_month]
        
        if not month_filtered.empty:
            st.subheader(f"Statistika za {selected_month}")
            
            # Priprava podatkov za stolpčni grafikon
            chart_data = month_filtered.groupby('Sentiment').agg(
                Število=('Sentiment', 'count'),
                Povprečno_Zaupanje=('Confidence', 'mean')
            ).reset_index()

            # Izris stolpčnega grafikona
            fig_month_bar = px.bar(
                chart_data, 
                x='Sentiment', 
                y='Število',
                color='Sentiment',
                custom_data=['Povprečno_Zaupanje'],
                color_discrete_map={'Pozitivno':'#2ca02c', 'Negativno':'#d62728'},
                text_auto=True
            )

            # Oblikovanje Tooltip-a (prikaz povprečnega zaupanja ob lebdenju)
            fig_month_bar.update_traces(
                hovertemplate="<br>".join([
                    "Sentiment: %{x}",
                    "Število: %{y}",
                    "Povprečno zaupanje modela: %{customdata[0]:.2%}"
                ])
            )

            st.plotly_chart(fig_month_bar, use_container_width=True)
            
            st.write(f"Pregled vseh {len(month_filtered)} ocen za {selected_month}:")
            st.dataframe(month_filtered[['date', 'review_text', 'Sentiment', 'Confidence']], use_container_width=True)
    else:
        st.warning("V podatkih ni najdenih ocen za leto 2023.")

elif page == "Izjave strank":
    st.header("💬 Izjave naših uporabnikov")
    if 'rating' in testimonials_df.columns:
        avg_rating = testimonials_df['rating'].mean()
        st.metric("Skupno zadovoljstvo", f"{avg_rating:.1f} / 5.0", "⭐")
    
    st.divider()
    for _, row in testimonials_df.iterrows():
        rating_val = row['rating'] if 'rating' in row else 5
        stars = "⭐" * int(rating_val) if rating_val > 0 else "⭐⭐⭐⭐⭐"
        with st.container():
            st.markdown(f"**{row['author']}** {stars}")
            st.write(f"*{row['text']}*")
            st.divider()

elif page == "Katalog izdelkov":
    st.header("📦 Katalog izdelkov")
    for _, row in products_df.iterrows():
        col1, col2 = st.columns([1, 3])
        with col1:
            if 'Slika_URL' in row and pd.notna(row['Slika_URL']):
                st.image(row['Slika_URL'], width=150)
            else:
                st.write("Ni slike")
        with col2:
            st.subheader(row['Ime izdelka'])
            st.write(f"**Cena:** {row['Cena']}")
            if 'Opis' in row:
                st.info(row['Opis'])
        st.divider()

st.sidebar.markdown("---")
st.sidebar.info("Model: DistilBERT (Deep Learning)")