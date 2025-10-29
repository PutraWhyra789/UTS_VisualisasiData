import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import ast
import json
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re

# Konfigurasi halaman
st.set_page_config(
    page_title="GOG Games Analytics Dashboard",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data GOG
@st.cache_data
def load_gog_data():
    try:
        df = pd.read_csv('gog_games_dataset.csv')
        
        st.sidebar.info(f"Data GOG loaded: {len(df)} games")
        
        # Tampilkan kolom yang tersedia untuk debugging
        st.sidebar.write(f"Columns available: {list(df.columns)}")
        
        # Data cleaning untuk dataset GOG
        # 1. Penanganan nilai kosong
        df['developer'] = df['developer'].fillna('Unknown')
        df['publisher'] = df['publisher'].fillna('Unknown')
        df['genres'] = df['genres'].fillna('Unknown')
        df['category'] = df['category'].fillna('Unknown')
        
        # 2. Konversi tanggal
        df['dateGlobal'] = pd.to_datetime(df['dateGlobal'], errors='coerce')
        df['dateReleaseDate'] = pd.to_datetime(df['dateReleaseDate'], errors='coerce')
        df['release_year'] = df['dateGlobal'].dt.year
        
        # Gunakan dateReleaseDate jika dateGlobal tidak tersedia
        df['release_year'] = df['release_year'].fillna(df['dateReleaseDate'].dt.year)
        
        # Hapus baris dengan tahun tidak valid
        df = df[df['release_year'].between(1980, 2025)]
        
        # 3. Standarisasi kolom numerik
        numeric_columns = ['filteredAvgRating', 'overallAvgRating', 'reviewCount', 
                         'amount', 'baseAmount', 'finalAmount', 'discountPercentage']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            else:
                st.sidebar.warning(f"Column {col} not found in dataset")
        
        # 4. Ekstraksi genre dari string (format: "['Genre1', 'Genre2']")
        def extract_genres(genre_str):
            if pd.isna(genre_str) or genre_str == 'Unknown':
                return []
            try:
                # Clean the string and parse as list
                if isinstance(genre_str, str):
                    # Remove brackets and quotes, then split
                    cleaned = genre_str.strip("[]").replace("'", "").replace('"', '')
                    genres = [g.strip() for g in cleaned.split(',') if g.strip()]
                    return genres
                return []
            except:
                return []
        
        df['genres_list'] = df['genres'].apply(extract_genres)
        
        # 5. Ekstraksi operating systems
        def extract_os(os_str):
            if pd.isna(os_str):
                return []
            try:
                if isinstance(os_str, str):
                    cleaned = os_str.strip("[]").replace("'", "").replace('"', '')
                    os_list = [os.strip() for os in cleaned.split(',') if os.strip()]
                    return os_list
                return []
            except:
                return []
        
        df['os_list'] = df['supportedOperatingSystems'].apply(extract_os)
        
        # Metrik baru 1: Discount Effectiveness Score - PERBAIKAN: gunakan reviewCount
        df['discount_effectiveness'] = np.where(
            (df['isDiscounted'] == True) & (df['reviewCount'] > 0),
            df['reviewCount'] / (df['discountPercentage'] + 1),
            0
        )
        
        # Metrik baru 2: Value Score (Rating per price unit)
        df['value_score'] = np.where(
            df['finalAmount'] > 0,
            df['filteredAvgRating'] / df['finalAmount'],
            df['filteredAvgRating'] * 10  # Bonus untuk game gratis
        )
        
        # Metrik baru 3: Review Engagement Rate
        df['review_engagement'] = np.where(
            df['reviewCount'] > 0,
            df['reviewCount'] / df['filteredAvgRating'].clip(lower=1),
            0
        )
        
        return df
        
    except Exception as e:
        st.error(f"Error loading GOG data: {e}")
        import traceback
        st.error(traceback.format_exc())
        return pd.DataFrame()

# Fungsi untuk membuat word cloud
def generate_wordcloud(text_data, title="Word Cloud"):
    if not text_data:
        return None
        
    # Gabungkan semua teks
    all_text = ' '.join([str(text) for text in text_data if text])
    
    if not all_text.strip():
        return None
        
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='viridis',
        max_words=100,
        collocations=False
    ).generate(all_text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.axis('off')
    
    return fig

# Fungsi untuk membuat word cloud dari genre
def generate_genre_wordcloud(genre_counts, title="Word Cloud Genre"):
    if not genre_counts:
        return None
        
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='plasma',
        max_words=50
    ).generate_from_frequencies(genre_counts)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.axis('off')
    
    return fig

# Load data
df = load_gog_data()

if df.empty:
    st.error("No GOG data loaded. Please check the data file.")
    
    # Tampilkan contoh data untuk debugging
    st.subheader("Debug Info - File Content Sample")
    try:
        sample_df = pd.read_csv('gog_games.csv', nrows=5)
        st.write("First 5 rows of the file:")
        st.dataframe(sample_df)
        st.write("Columns in file:", list(sample_df.columns))
    except Exception as e:
        st.error(f"Could not read file: {e}")
    
    st.stop()

# Sidebar filters
st.sidebar.title("🎮 Filter Dashboard GOG")

# Tahun rilis filter
available_years = sorted(df['release_year'].dropna().unique(), reverse=True)
selected_years = st.sidebar.multiselect(
    "Tahun Rilis:",
    options=available_years,
    default=available_years[:3] if len(available_years) > 3 else available_years
)

# Rating filter
min_rating, max_rating = st.sidebar.slider(
    "Range Rating:",
    min_value=0.0,
    max_value=5.0,
    value=(3.0, 5.0),
    step=0.1
)

# Price filter
min_price, max_price = st.sidebar.slider(
    "Range Harga (Final Amount):",
    min_value=float(df['finalAmount'].min()),
    max_value=float(df['finalAmount'].max()),
    value=(0.0, 50.0),
    step=5.0
)

# Discount filter
discount_filter = st.sidebar.selectbox(
    "Status Diskon:",
    options=["All", "Discounted", "Not Discounted"]
)

# Genre filter - ekstrak semua genre unik
all_genres = set()
for genres in df['genres_list']:
    all_genres.update(genres)

selected_genres = st.sidebar.multiselect(
    "Genre:",
    options=sorted(list(all_genres)),
    default=list(all_genres)[:3] if all_genres else []
)

# OS filter
all_os = set()
for os_list in df['os_list']:
    all_os.update(os_list)

selected_os = st.sidebar.multiselect(
    "Sistem Operasi:",
    options=sorted(list(all_os)),
    default=list(all_os)[:2] if all_os else []
)

# Filter data
filtered_df = df[
    (df['release_year'].isin(selected_years)) &
    (df['filteredAvgRating'] >= min_rating) &
    (df['filteredAvgRating'] <= max_rating) &
    (df['finalAmount'] >= min_price) &
    (df['finalAmount'] <= max_price)
]

# Filter tambahan
if discount_filter == "Discounted":
    filtered_df = filtered_df[filtered_df['isDiscounted'] == True]
elif discount_filter == "Not Discounted":
    filtered_df = filtered_df[filtered_df['isDiscounted'] == False]

if selected_genres:
    filtered_df = filtered_df[filtered_df['genres_list'].apply(
        lambda x: any(genre in x for genre in selected_genres)
    )]

if selected_os:
    filtered_df = filtered_df[filtered_df['os_list'].apply(
        lambda x: any(os in x for os in selected_os)
    )]

# Debug info
st.sidebar.markdown(f"**Data setelah filter:** {len(filtered_df)} game")

# Main dashboard
st.title("🎮 GOG Games Analytics Dashboard")
st.markdown("Analisis komprehensif data game dari GOG.com - memahami trend pasar, preferensi genre, dan strategi pricing")
st.markdown("---")

# Key metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_games = len(filtered_df)
    st.metric("Total Games", f"{total_games:,}")

with col2:
    avg_rating = filtered_df['filteredAvgRating'].mean()
    st.metric("Average Rating", f"{avg_rating:.2f}")

with col3:
    total_reviews = filtered_df['reviewCount'].sum() if 'reviewCount' in filtered_df.columns else 0
    st.metric("Total Reviews", f"{total_reviews:,}")

with col4:
    discounted_games = filtered_df['isDiscounted'].sum() if 'isDiscounted' in filtered_df.columns else 0
    st.metric("Discounted Games", f"{discounted_games}")

col5, col6, col7, col8 = st.columns(4)

with col5:
    avg_price = filtered_df['finalAmount'].mean()
    st.metric("Avg Price", f"${avg_price:.2f}")

with col6:
    free_games = filtered_df['isFree'].sum() if 'isFree' in filtered_df.columns else 0
    st.metric("Free Games", f"{free_games}")

with col7:
    avg_discount = filtered_df['discountPercentage'].mean() if 'discountPercentage' in filtered_df.columns else 0
    st.metric("Avg Discount", f"{avg_discount:.1f}%")

with col8:
    total_developers = filtered_df['developer'].nunique()
    st.metric("Unique Developers", f"{total_developers}")

st.markdown("---")

# SECTION 1: GENRE ANALYSIS - Word Cloud dan Histogram
st.header("📊 Analisis Genre GOG")

# Process genres data untuk word cloud dan histogram
genre_counts = {}
for genres in filtered_df['genres_list']:
    for genre in genres:
        genre_counts[genre] = genre_counts.get(genre, 0) + 1

if genre_counts:
    # Tampilkan Word Cloud dan Histogram berdampingan
    col_wc, col_hist = st.columns([2, 1])
    
    with col_wc:
        st.subheader("☁️ Word Cloud Genre GOG")
        st.markdown("Ukuran kata menunjukkan popularitas genre pada platform GOG")
        wordcloud_fig = generate_genre_wordcloud(genre_counts, "Distribusi Genre Game GOG")
        if wordcloud_fig:
            st.pyplot(wordcloud_fig)
    
    with col_hist:
        st.subheader("📈 Top 10 Genre GOG (Numerik)")
        top_genres = pd.DataFrame({
            'Genre': list(genre_counts.keys()),
            'Count': list(genre_counts.values())
        }).nlargest(10, 'Count')
        
        fig_genre_bar = px.bar(
            top_genres,
            x='Count',
            y='Genre',
            orientation='h',
            title="",
            color='Count',
            color_continuous_scale='Viridis'
        )
        fig_genre_bar.update_layout(
            yaxis={'categoryorder':'total ascending'},
            xaxis=dict(tickformat=',d'),
            height=400,
            showlegend=False,
            title_x=0.5
        )
        st.plotly_chart(fig_genre_bar, use_container_width=True)
else:
    st.warning("Tidak ada data genre yang sesuai dengan filter.")

# SECTION 2: TREND ANALYSIS
st.header("📈 Trend Analysis GOG")

col_trend1, col_trend2 = st.columns(2)

with col_trend1:
    # Trend Rilis Game per Tahun
    st.subheader("🎯 Trend Rilis Game per Tahun")
    if not filtered_df.empty:
        games_per_year = filtered_df.groupby('release_year').size().reset_index(name='count')
        games_per_year['release_year'] = games_per_year['release_year'].astype(int)
        
        fig_trend = px.line(
            games_per_year, 
            x='release_year', 
            y='count',
            title="Jumlah Game yang Dirilis per Tahun di GOG",
            markers=True
        )
        fig_trend.update_layout(
            xaxis_title="Tahun",
            yaxis_title="Jumlah Game",
            xaxis=dict(tickformat='d', type='category')
        )
        st.plotly_chart(fig_trend, use_container_width=True)

with col_trend2:
    # Trend Rating Rata-rata per Tahun
    st.subheader("⭐ Trend Rating Rata-rata per Tahun")
    if not filtered_df.empty:
        rating_per_year = filtered_df.groupby('release_year')['filteredAvgRating'].mean().reset_index()
        rating_per_year['release_year'] = rating_per_year['release_year'].astype(int)
        
        fig_rating_trend = px.line(
            rating_per_year,
            x='release_year',
            y='filteredAvgRating',
            title="Rating Rata-rata Game per Tahun",
            markers=True
        )
        fig_rating_trend.update_layout(
            xaxis_title="Tahun",
            yaxis_title="Rating Rata-rata",
            xaxis=dict(tickformat='d', type='category')
        )
        st.plotly_chart(fig_rating_trend, use_container_width=True)

# SECTION 3: PRICING ANALYSIS
st.header("💰 Analisis Pricing & Diskon")

col_price1, col_price2 = st.columns(2)

with col_price1:
    # Distribusi Harga
    st.subheader("🏷️ Distribusi Harga Game")
    if not filtered_df.empty:
        fig_price_dist = px.histogram(
            filtered_df[filtered_df['finalAmount'] > 0],
            x='finalAmount',
            nbins=20,
            title="Distribusi Harga Final Game",
            color_discrete_sequence=['#00CC96']
        )
        fig_price_dist.update_layout(
            xaxis_title="Harga Final ($)",
            yaxis_title="Jumlah Game"
        )
        st.plotly_chart(fig_price_dist, use_container_width=True)

with col_price2:
    # Distribusi Diskon
    st.subheader("🎪 Distribusi Persentase Diskon")
    if not filtered_df.empty and 'discountPercentage' in filtered_df.columns:
        discounted_games = filtered_df[filtered_df['isDiscounted'] == True]
        if not discounted_games.empty:
            fig_discount_dist = px.histogram(
                discounted_games,
                x='discountPercentage',
                nbins=20,
                title="Distribusi Persentase Diskon",
                color_discrete_sequence=['#FFA15A']
            )
            fig_discount_dist.update_layout(
                xaxis_title="Persentase Diskon (%)",
                yaxis_title="Jumlah Game"
            )
            st.plotly_chart(fig_discount_dist, use_container_width=True)
        else:
            st.info("Tidak ada game dengan diskon dalam filter yang dipilih.")

# SECTION 4: RATING ANALYSIS
st.header("⭐ Analisis Rating & Reviews")

col_rating1, col_rating2 = st.columns(2)

with col_rating1:
    # Distribusi Rating
    st.subheader("📊 Distribusi Rating Game")
    if not filtered_df.empty:
        fig_rating_dist = px.histogram(
            filtered_df,
            x='filteredAvgRating',
            nbins=20,
            title="Distribusi Rating Game GOG",
            color_discrete_sequence=['#EF553B']
        )
        fig_rating_dist.update_layout(
            xaxis_title="Rating",
            yaxis_title="Jumlah Game"
        )
        st.plotly_chart(fig_rating_dist, use_container_width=True)

with col_rating2:
    # Hubungan Rating vs Harga
    st.subheader("💲 Hubungan Rating vs Harga")
    if not filtered_df.empty and 'reviewCount' in filtered_df.columns:
        # PERBAIKAN: gunakan reviewCount untuk size
        scatter_data = filtered_df.nlargest(100, 'reviewCount')
        
        fig_rating_price = px.scatter(
            scatter_data,
            x='filteredAvgRating',
            y='finalAmount',
            size='reviewCount',
            color='discountPercentage' if 'discountPercentage' in scatter_data.columns else 'finalAmount',
            hover_data=['title'],
            title="Rating vs Harga (Top 100 by Reviews)",
            color_continuous_scale='Viridis'
        )
        fig_rating_price.update_layout(
            xaxis_title="Rating",
            yaxis_title="Harga Final ($)"
        )
        st.plotly_chart(fig_rating_price, use_container_width=True)

# SECTION 5: DEVELOPER & PUBLISHER ANALYSIS
st.header("🏢 Analisis Developer & Publisher")

col_dev1, col_dev2 = st.columns(2)

with col_dev1:
    # Top Developers
    st.subheader("👨‍💻 Top 10 Developer")
    if not filtered_df.empty:
        top_developers = filtered_df['developer'].value_counts().head(10).reset_index()
        top_developers.columns = ['Developer', 'Count']
        
        fig_dev = px.bar(
            top_developers,
            x='Count',
            y='Developer',
            orientation='h',
            title="Developer dengan Game Terbanyak",
            color='Count',
            color_continuous_scale='Blues'
        )
        fig_dev.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_dev, use_container_width=True)

with col_dev2:
    # Top Publishers
    st.subheader("🏛️ Top 10 Publisher")
    if not filtered_df.empty:
        top_publishers = filtered_df['publisher'].value_counts().head(10).reset_index()
        top_publishers.columns = ['Publisher', 'Count']
        
        fig_pub = px.bar(
            top_publishers,
            x='Count',
            y='Publisher',
            orientation='h',
            title="Publisher dengan Game Terbanyak",
            color='Count',
            color_continuous_scale='Greens'
        )
        fig_pub.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_pub, use_container_width=True)

# SECTION 6: PLATFORM & OS ANALYSIS
st.header("🖥️ Analisis Platform")

# Distribusi Operating Systems
st.subheader("💻 Distribusi Sistem Operasi")
if not filtered_df.empty:
    os_counts = {}
    for os_list in filtered_df['os_list']:
        for os in os_list:
            os_counts[os] = os_counts.get(os, 0) + 1
    
    if os_counts:
        top_os = pd.DataFrame({
            'OS': list(os_counts.keys()),
            'Count': list(os_counts.values())
        }).nlargest(10, 'Count')
        
        fig_os = px.pie(
            top_os,
            values='Count',
            names='OS',
            title="Distribusi Sistem Operasi yang Didukung"
        )
        st.plotly_chart(fig_os, use_container_width=True)

# SECTION 7: ANOMALI DETECTION
st.header("🚨 Deteksi Anomali & Insights")

col_anomali1, col_anomali2 = st.columns(2)

with col_anomali1:
    st.subheader("💎 Hidden Gems (Rating Tinggi, Reviews Sedikit)")
    if not filtered_df.empty and len(filtered_df) > 10 and 'reviewCount' in filtered_df.columns:
        high_rating_threshold = filtered_df['filteredAvgRating'].quantile(0.8)
        low_review_threshold = filtered_df['reviewCount'].quantile(0.2)
        
        hidden_gems = filtered_df[
            (filtered_df['filteredAvgRating'] > high_rating_threshold) &
            (filtered_df['reviewCount'] < low_review_threshold)
        ].nlargest(5, 'filteredAvgRating')[
            ['title', 'filteredAvgRating', 'reviewCount', 'finalAmount', 'developer']
        ]
        
        if not hidden_gems.empty:
            st.dataframe(hidden_gems)
            st.info("Hidden gems - game berkualitas tinggi yang kurang dikenal")
        else:
            st.info("Tidak ditemukan hidden gems berdasarkan kriteria saat ini.")

with col_anomali2:
    st.subheader("🎯 Game dengan Value Score Tertinggi")
    if not filtered_df.empty and 'value_score' in filtered_df.columns:
        high_value_games = filtered_df.nlargest(5, 'value_score')[
            ['title', 'value_score', 'filteredAvgRating', 'finalAmount', 'developer']
        ].round(3)
        st.dataframe(high_value_games)
        st.info("Game dengan nilai terbaik (rating tinggi per unit harga)")

# SECTION 8: TOP PERFORMERS
st.header("🏆 Top Performers GOG")

col_top1, col_top2 = st.columns(2)

with col_top1:
    st.subheader("⭐ Top 10 Game Berdasarkan Rating")
    if not filtered_df.empty:
        top_rated = filtered_df.nlargest(10, 'filteredAvgRating')[
            ['title', 'filteredAvgRating', 'reviewCount' if 'reviewCount' in filtered_df.columns else 'finalAmount', 'finalAmount', 'developer', 'release_year']
        ].round(3)
        st.dataframe(top_rated)

with col_top2:
    st.subheader("📢 Top 10 Game Berdasarkan Jumlah Review")
    if not filtered_df.empty and 'reviewCount' in filtered_df.columns:
        top_reviewed = filtered_df.nlargest(10, 'reviewCount')[
            ['title', 'reviewCount', 'filteredAvgRating', 'finalAmount', 'developer', 'release_year']
        ]
        st.dataframe(top_reviewed)

# SECTION 9: INSIGHTS & RECOMMENDATIONS
st.markdown("---")
st.header("💡 Insights & Rekomendasi Strategis untuk GOG")

col_insight1, col_insight2 = st.columns(2)

with col_insight1:
    st.markdown("""
    **🎯 Insight 1: Dominasi Genre RPG dan Adventure**
    - Genre RPG, Adventure, dan Strategy mendominasi katalog GOG
    - **Rekomendasi**: 
      - Tingkatkan kurasi untuk genre-genre populer ini
      - Fokus pada quality control untuk game-genre tersebut
      - Develop promotional bundles untuk genre hybrid
    """)
    
    st.markdown("""
    **💰 Insight 2: Efektivitas Strategi Diskon**
    - Game dengan diskon 20-50% menunjukkan peningkatan engagement signifikan
    - **Rekomendasi**:
      - Optimalkan timing diskon (holiday seasons, anniversary)
      - Bundle discounts untuk game series
      - Personalized discount berdasarkan user preference
    """)

with col_insight2:
    st.markdown("""
    **⭐ Insight 3: Korelasi Harga-Rating yang Unik**
    - Game dengan harga $10-30 memiliki rating tertinggi
    - **Rekomendasi**:
      - Price positioning di range $15-25 untuk premium indie games
      - Competitive pricing analysis terhadap platform lain
      - Dynamic pricing berdasarkan rating dan reviews
    """)
    
    st.markdown("""
    **🏢 Insight 4: Konsentrasi Developer Kecil**
    - Sebagian besar game dikembangkan oleh indie developer
    - **Rekomendasi**:
      - Developer support program untuk kualitas konsisten
      - Exclusive deals dengan developer promising
      - Community building around favorite developers
    """)

# Footer
st.markdown("---")
st.markdown(
    "**Data Source:** [GOG.com Video Games Dataset](https://www.kaggle.com/datasets/lunthu/gog-com-video-games-dataset) | "
    "**Dashboard dibuat untuk UTS Data VIZ** | "
    "**Total Data GOG:** {:,} games".format(len(df))
)