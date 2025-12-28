import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# 1. Nastavitve strani
st.set_page_config(page_title="Brand Sentiment Analysis 2023", layout="wide")

# 2. Hitro nalaganje modela
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

sentiment_pipeline = load_sentiment_model()

# 3. Optimizirana masovna AI analiza
@st.cache_data
def get_bulk_sentiment(texts):
    processed_texts = [str(t)[:512] for t in texts if pd.notna(t)]
    if not processed_texts:
        return [], []
    results = sentiment_pipeline(processed_texts)
    sentiments = ["Pozitivno" if r['label'] == 'POSITIVE' else "Negativno" for r in results]
    confidences = [round(r['score'], 3) for r in results]
    return sentiments, confidences

# 4. Funkcija za Word Cloud (DODANO NAZAJ)
def prikazi_dvojni_wordcloud(df_mesec):
    pozitivna = df_mesec[df_mesec['Sentiment'] == 'Pozitivno']['review_text']
    negativna = df_mesec[df_mesec['Sentiment'] == 'Negativno']['review_text']
    
    stop_besede = {'in', 'je', 'da', 'pa', 'za', 'na', 'bi', 'se', 'iz', 'the', 'and', 'was', 'with', 'to', 'for', 'it', 'of', 'i', 'you', 'my', 'is', 'a'}
    
    col_wc1, col_wc2 = st.columns(2)
    with col_wc1:
        st.write("✅ **Pozitivno**")
        if not pozitivna.empty and pozitivna.str.cat(sep=' ').strip():
            wc_pos = WordCloud(width=400, height=300, background_color='white', colormap='Greens', stopwords=stop_besede).generate(" ".join(pozitivna.astype(str)))
            fig_pos, ax_pos = plt.subplots()
            ax_pos.imshow(wc_pos, interpolation='bilinear')
            ax_pos.axis("off")
            st.pyplot(fig_pos)
        else:
            st.info("Ni dovolj pozitivnih besed.")

    with col_wc2:
        st.write("❌ **Negativno**")
        if not negativna.empty and negativna.str.cat(sep=' ').strip():
            wc_neg = WordCloud(width=400, height=300, background_color='white', colormap='Reds', stopwords=stop_besede).generate(" ".join(negativna.astype(str)))
            fig_neg, ax_neg = plt.subplots()
            ax_neg.imshow(wc_neg, interpolation='bilinear')
            ax_neg.axis("off")
            st.pyplot(fig_neg)
        else:
            st.info("Ni dovolj negativnih besed.")

# 5. Nalaganje podatkov
@st.cache_data
def load_data():
    try:
        rev_df = pd.read_csv("reviews.csv")
        rev_df['date'] = pd.to_datetime(rev_df['date'])
        test_df = pd.read_csv("testimonials.csv")
        prod_df = pd.read_csv("products.csv")
        return rev_df, test_df, prod_df
    except:
        return None, None, None

reviews_df, testimonials_df, products_df = load_data()

# 6. Navigacija
page = st.sidebar.radio("Izberi pogled:", ["Analiza sentimenta", "Izjave strank", "Katalog izdelkov"])

if reviews_df is not None:
    df_2023 = reviews_df[reviews_df['date'].dt.year == 2023].copy()
    df_2023['Month'] = df_2023['date'].dt.month_name()
    month_order = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
else:
    df_2023 = pd.DataFrame()

st.title("📈 Brand Sentiment Analysis 2023")

if page == "Analiza sentimenta":
    if not df_2023.empty:
        with st.spinner('Analiziram mnenja...'):
            sentiments, confidences = get_bulk_sentiment(df_2023['review_text'].tolist())
            df_2023['Sentiment'] = sentiments
            df_2023['Confidence'] = confidences

        # Grafi
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.pie(df_2023, names='Sentiment', color='Sentiment', 
                                   color_discrete_map={'Pozitivno':'#2ca02c', 'Negativno':'#d62728'}, 
                                   title="Skupni sentiment"), use_container_width=True)
        with c2:
            st.plotly_chart(px.histogram(df_2023, x='Month', color='Sentiment', barmode='group',
                                          category_orders={"Month": month_order}, 
                                          title="Mesečni trend"), use_container_width=True)

        st.divider()
        available_months = [m for m in month_order if m in df_2023['Month'].unique()]
        selected_month = st.select_slider("Izberi mesec za Word Cloud:", options=available_months)
        
        month_filtered = df_2023[df_2023['Month'] == selected_month]
        prikazi_dvojni_wordcloud(month_filtered)
        
        st.subheader(f"Podatki za {selected_month}")
        st.dataframe(month_filtered[['date', 'review_text', 'Sentiment', 'Confidence']], use_container_width=True, hide_index=True)

elif page == "Izjave strank":
    st.header("💬 Izjave strank")
    if testimonials_df is not None:
        st.dataframe(testimonials_df, use_container_width=True, hide_index=True)

elif page == "Katalog izdelkov":
    st.header("📦 Katalog izdelkov")
    if products_df is not None:
        # Pokažemo le besedilne stolpce brez slik
        prikaz_cols = [c for c in ['Ime izdelka', 'Cena', 'Opis'] if c in products_df.columns]
        st.dataframe(products_df[prikaz_cols], use_container_width=True, hide_index=True)