import os
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Zomato Analytics & Clustering",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Theme: Midnight navy + teal ----------
st.markdown("""
<style>
:root {
    --app-bg: #0B1220;
    --sidebar-bg: #101B2D;
    --panel-bg: #16243A;
    --panel-bg-2: #1B2D46;
    --border: #2B405D;
    --text-main: #F3F7FC;
    --text-muted: #B8C6D9;
    --accent: #2DD4BF;
    --accent-2: #60A5FA;
}
html, body, [class*="css"] { font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
.stApp { background: var(--app-bg); color: var(--text-main); }
[data-testid="stHeader"] { background: rgba(11,18,32,.96); border-bottom: 1px solid #1E2C42; }
[data-testid="stToolbar"] { right: 1rem; }
[data-testid="stSidebar"] { background: var(--sidebar-bg); border-right: 1px solid var(--border); }
[data-testid="stSidebar"] * { color: var(--text-main) !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stCaption { color: var(--text-muted) !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] label { padding: 7px 10px; border-radius: 9px; }
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background: #1B2D46; }
h1, h2, h3, h4 { color: var(--text-main) !important; letter-spacing: -0.025em; }
h1 { font-weight: 750 !important; }
p, li, label, .stMarkdown, [data-testid="stCaptionContainer"] { color: var(--text-muted); }
.subtitle { color: #9FE7DE !important; font-size: 1.02rem; margin-top: -.35rem; }
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #1A2B43, #142238);
    border: 1px solid var(--border); border-radius: 14px;
    padding: 18px 20px; box-shadow: 0 8px 24px rgba(0,0,0,.14);
}
[data-testid="stMetricLabel"] { color: #B8C6D9 !important; font-size: .9rem !important; }
[data-testid="stMetricValue"] { color: #F8FAFC !important; font-weight: 750 !important; }
[data-testid="stMetricDelta"] { color: var(--accent) !important; }
[data-testid="stVerticalBlockBorderWrapper"] { border-color: var(--border) !important; }
[data-testid="stDataFrame"], [data-testid="stTable"] { border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }
.stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"],
.stMultiSelect div[data-baseweb="select"], .stSlider, .stFileUploader section {
    background: #17263D !important; color: var(--text-main) !important;
}
.stTextInput input, .stNumberInput input { border: 1px solid var(--border) !important; }
.stButton button, .stFormSubmitButton button {
    background: var(--accent) !important; color: #06201E !important;
    border: 0 !important; border-radius: 9px !important; font-weight: 700 !important;
}
.stButton button:hover, .stFormSubmitButton button:hover { background: #5EEAD4 !important; }
[data-testid="stAlert"] { border-radius: 12px; }
hr { border-color: var(--border) !important; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1500px; }
</style>
""", unsafe_allow_html=True)

# Use a consistent dark Plotly theme so charts never appear as white blocks.
px.defaults.template = "plotly_dark"
px.defaults.color_discrete_sequence = ["#2DD4BF", "#60A5FA", "#A78BFA", "#F59E0B", "#F472B6", "#34D399"]

def style_chart(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, sans-serif", color="#E5EDF7", size=13),
        title=dict(font=dict(size=18, color="#F8FAFC"), x=0.02, xanchor="left"),
        margin=dict(l=24, r=24, t=62, b=28),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#CBD5E1")),
        colorway=["#2DD4BF", "#60A5FA", "#A78BFA", "#F59E0B", "#F472B6", "#34D399"],
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,.16)", zerolinecolor="rgba(148,163,184,.2)",
                     linecolor="rgba(148,163,184,.25)", tickfont=dict(color="#B8C6D9"), title_font=dict(color="#E5EDF7"))
    fig.update_yaxes(gridcolor="rgba(148,163,184,.16)", zerolinecolor="rgba(148,163,184,.2)",
                     linecolor="rgba(148,163,184,.25)", tickfont=dict(color="#B8C6D9"), title_font=dict(color="#E5EDF7"))
    return fig

META_DEFAULT = "data/Zomato Restaurant names and Metadata.csv"
REV_DEFAULT = "data/Zomato Restaurant reviews.csv"

@st.cache_data(show_spinner=False)
def load_data(meta_bytes=None, rev_bytes=None):
    if meta_bytes is not None and rev_bytes is not None:
        from io import BytesIO
        metadata = pd.read_csv(BytesIO(meta_bytes))
        reviews = pd.read_csv(BytesIO(rev_bytes))
    else:
        if not (os.path.exists(META_DEFAULT) and os.path.exists(REV_DEFAULT)):
            raise FileNotFoundError(
                "Dataset files were not found. Put both CSV files inside the data/ folder "
                "or upload them using the sidebar."
            )
        metadata = pd.read_csv(META_DEFAULT)
        reviews = pd.read_csv(REV_DEFAULT)
    return metadata, reviews

def prepare_data(metadata, reviews):
    required_meta = {"Name", "Cost", "Cuisines"}
    required_reviews = {"Restaurant", "Review", "Rating", "Pictures"}
    if not required_meta.issubset(metadata.columns):
        raise ValueError(f"Metadata CSV must contain: {sorted(required_meta)}")
    if not required_reviews.issubset(reviews.columns):
        raise ValueError(f"Reviews CSV must contain: {sorted(required_reviews)}")

    meta = metadata.copy()
    rev = reviews.copy()

    for col in ["Collections", "Timings"]:
        if col in meta.columns:
            meta[col] = meta[col].fillna("Not Available")
    for col in ["Reviewer", "Metadata"]:
        if col in rev.columns:
            rev[col] = rev[col].fillna("Unknown")

    meta["Name"] = meta["Name"].astype(str).str.strip()
    rev["Restaurant"] = rev["Restaurant"].astype(str).str.strip()
    meta["Cost"] = pd.to_numeric(
        meta["Cost"].astype(str).str.replace(",", "", regex=False),
        errors="coerce"
    )
    rev["Rating"] = pd.to_numeric(rev["Rating"], errors="coerce")
    rev["Pictures"] = pd.to_numeric(rev["Pictures"], errors="coerce")

    if "Time" in rev.columns:
        rev["Time"] = pd.to_datetime(rev["Time"], errors="coerce")

    meta = meta.drop_duplicates().copy()
    rev = rev.drop_duplicates().copy()

    rev["restaurant_key"] = rev["Restaurant"].str.casefold().str.strip()
    meta["restaurant_key"] = meta["Name"].str.casefold().str.strip()

    # Matches the notebook's restaurant-level aggregation:
    # Review_Count counts non-null Review values.
    stats = rev.groupby("restaurant_key").agg(
        Average_Rating=("Rating", "mean"),
        Review_Count=("Review", "count"),
        Average_Pictures=("Pictures", "mean"),
        Total_Reviews=("Restaurant", "size")
    ).reset_index()

    data = meta.merge(stats, on="restaurant_key", how="left")
    data["Cuisine_Count"] = data["Cuisines"].fillna("").apply(
        lambda x: len([item for item in str(x).split(",") if item.strip()])
    )
    data["Review_Count"] = data["Review_Count"].fillna(0)
    data["Total_Reviews"] = data["Total_Reviews"].fillna(0)
    data["Average_Pictures"] = data["Average_Pictures"].fillna(0)
    data["Price_Category"] = pd.cut(
        data["Cost"], bins=[0, 500, 1500, np.inf],
        labels=["Budget", "Mid-Range", "Premium"], include_lowest=True
    )
    return meta, rev, data

def build_model_data(data):
    features = ["Cost", "Average_Rating", "Review_Count", "Cuisine_Count", "Average_Pictures"]
    model_df = data[features].copy()
    model_df = model_df.replace([np.inf, -np.inf], np.nan)
    medians = model_df.median(numeric_only=True)
    model_df = model_df.fillna(medians).fillna(0)

    transformed = model_df.copy()
    transformed["Cost"] = np.log1p(transformed["Cost"].clip(lower=0))
    transformed["Review_Count"] = np.log1p(transformed["Review_Count"].clip(lower=0))

    scaler = StandardScaler()
    scaled = scaler.fit_transform(transformed)
    scaled_df = pd.DataFrame(scaled, columns=features, index=data.index)

    scores = []
    candidate_ks = range(2, min(11, len(scaled_df)))
    for k in candidate_ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(scaled_df)
        scores.append((k, silhouette_score(scaled_df, labels)))
    if not scores:
        raise ValueError("At least three usable restaurants are required for clustering.")
    best_k = max(scores, key=lambda item: item[1])[0]
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(scaled_df)

    # Notebook's Agglomerative approach: Ward linkage, best k by silhouette.
    agg_candidates = []
    for k in range(2, min(11, len(scaled_df))):
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(scaled_df)
        agg_candidates.append((k, silhouette_score(scaled_df, labels), labels))
    best_agg = max(agg_candidates, key=lambda item: item[1])
    agg_labels = best_agg[2]

    # Notebook's fixed DBSCAN settings.
    dbscan = DBSCAN(eps=0.8, min_samples=4)
    dbscan_labels = dbscan.fit_predict(scaled_df)

    return model_df, transformed, scaler, scaled_df, kmeans, kmeans_labels, best_k, scores, agg_labels, best_agg[0], dbscan_labels

def metric_values(X, labels, ignore_noise=False):
    labels = np.asarray(labels)
    mask = labels != -1 if ignore_noise else np.ones(len(labels), dtype=bool)
    y = labels[mask]
    x = np.asarray(X)[mask]
    n_clusters = len(set(y))
    if n_clusters < 2 or n_clusters >= len(y):
        return np.nan, np.nan, np.nan
    return (
        silhouette_score(x, y),
        davies_bouldin_score(x, y),
        calinski_harabasz_score(x, y)
    )

def make_chart(choice, d, reviews, cluster_col):
    rating_data = reviews.dropna(subset=["Rating"]).copy()
    chart = None

    if choice == "1. Restaurant cost distribution":
        chart = px.histogram(d.dropna(subset=["Cost"]), x="Cost", nbins=20, marginal="box",
                             title="Restaurant Cost Distribution", labels={"Cost":"Cost for two (₹)"})
    elif choice == "2. Customer rating distribution":
        chart = px.histogram(rating_data, x="Rating", nbins=10, title="Customer Rating Distribution")
    elif choice == "3. Top cuisines":
        cuisines = d["Cuisines"].dropna().str.split(",").explode().str.strip()
        top = cuisines[cuisines.ne("")].value_counts().head(15).rename_axis("Cuisine").reset_index(name="Restaurants")
        chart = px.bar(top.sort_values("Restaurants"), x="Restaurants", y="Cuisine", orientation="h",
                       title="Top 15 Cuisines")
    elif choice == "4. Cost vs average rating":
        chart = px.scatter(d, x="Cost", y="Average_Rating", hover_name="Name",
                           title="Restaurant Cost vs Average Rating", labels={"Cost":"Cost for two (₹)"})
    elif choice == "5. Cuisine count vs average rating":
        chart = px.scatter(d, x="Cuisine_Count", y="Average_Rating", hover_name="Name",
                           title="Cuisine Count vs Average Rating")
    elif choice == "6. Customer rating boxplot":
        chart = px.box(rating_data, y="Rating", points="outliers", title="Customer Rating Boxplot")
    elif choice == "7. Cost across top 5 cuisines":
        top_cuisines = d["Cuisines"].dropna().str.split(",").explode().str.strip().value_counts().head(5).index
        temp = d[["Name","Cost","Cuisines"]].dropna(subset=["Cost"]).copy()
        temp["Cuisine"] = temp["Cuisines"].fillna("").str.split(",")
        temp = temp.explode("Cuisine")
        temp["Cuisine"] = temp["Cuisine"].str.strip()
        temp = temp[temp["Cuisine"].isin(top_cuisines)]
        chart = px.box(temp, x="Cuisine", y="Cost", title="Cost Distribution Across Top 5 Cuisines")
    elif choice == "8. Reviews per restaurant":
        counts = reviews["Restaurant"].value_counts().reset_index()
        counts.columns = ["Restaurant","Review_Count"]
        chart = px.histogram(counts, x="Review_Count", nbins=20, title="Distribution of Reviews per Restaurant")
    elif choice == "9. Average pictures vs review count":
        chart = px.scatter(d, x="Review_Count", y="Average_Pictures", hover_name="Name",
                           title="Average Pictures vs Review Count")
    elif choice == "10. Average rating by price category":
        chart = px.box(d.dropna(subset=["Price_Category","Average_Rating"]),
                       x="Price_Category", y="Average_Rating", title="Average Rating by Price Category")
    elif choice == "11. Top 10 restaurants by review count":
        top = d.nlargest(10, "Review_Count")
        chart = px.bar(top.sort_values("Review_Count"), x="Review_Count", y="Name", orientation="h",
                       title="Top 10 Restaurants by Review Count")
    elif choice == "12. Rating vs pictures":
        chart = px.box(rating_data.dropna(subset=["Pictures"]), x="Rating", y="Pictures",
                       title="Customer Rating vs Pictures")
    elif choice == "13. Average rating vs review count":
        chart = px.scatter(d, x="Review_Count", y="Average_Rating", hover_name="Name",
                           title="Average Rating vs Review Count")
    elif choice == "14. Correlation heatmap":
        cols = ["Cost","Average_Rating","Review_Count","Cuisine_Count","Average_Pictures"]
        corr = d[cols].corr(numeric_only=True)
        chart = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r",
                          zmin=-1, zmax=1, title="Feature Correlation Heatmap")
    elif choice == "15. Pair plot":
        cols = ["Cost","Average_Rating","Review_Count","Cuisine_Count","Average_Pictures"]
        pair_df = d[cols].dropna()
        if len(pair_df) > 300:
            pair_df = pair_df.sample(300, random_state=42)
        chart = px.scatter_matrix(pair_df, dimensions=cols, title="Pair Plot of Restaurant Features")
    elif choice == "Cluster distribution":
        counts = d[cluster_col].astype(str).value_counts().rename_axis("Cluster").reset_index(name="Restaurants")
        chart = px.pie(counts, names="Cluster", values="Restaurants", title=f"{cluster_col} Distribution")
    elif choice == "Cluster feature profile":
        cols = ["Cost","Average_Rating","Review_Count","Cuisine_Count","Average_Pictures"]
        prof = d.groupby(cluster_col)[cols].mean().reset_index()
        long = prof.melt(id_vars=cluster_col, var_name="Feature", value_name="Mean Value")
        chart = px.bar(long, x="Feature", y="Mean Value", color=cluster_col, barmode="group",
                       title=f"{cluster_col} - Average Feature Profile")
    return chart

# ---------- Sidebar ----------
st.sidebar.title("🍽️ Zomato Insights")
st.sidebar.caption("Restaurant analytics & machine learning")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "EDA Chart Explorer", "Machine Learning", "Restaurant Explorer", "Cluster Prediction"]
)
st.sidebar.divider()
st.sidebar.subheader("Dataset source")
meta_upload = st.sidebar.file_uploader("Upload metadata CSV (optional)", type=["csv"], key="meta")
rev_upload = st.sidebar.file_uploader("Upload reviews CSV (optional)", type=["csv"], key="reviews")

try:
    metadata, reviews = load_data(
        meta_upload.getvalue() if meta_upload else None,
        rev_upload.getvalue() if rev_upload else None
    )
    metadata_clean, reviews_clean, restaurant_df = prepare_data(metadata, reviews)
    model_df, transformed_df, scaler, scaled_df, kmeans_model, kmeans_labels, best_k, k_scores, agg_labels, agg_k, dbscan_labels = build_model_data(restaurant_df)
except Exception as exc:
    st.error(str(exc))
    st.info("Expected local files: data/Zomato Restaurant names and Metadata.csv and data/Zomato Restaurant reviews.csv")
    st.stop()

restaurant_df = restaurant_df.copy()
restaurant_df["KMeans_Cluster"] = kmeans_labels
restaurant_df["Agglomerative_Cluster"] = agg_labels
restaurant_df["DBSCAN_Cluster"] = dbscan_labels

# ---------- Header ----------
st.title("🍽️ Zomato Restaurant Analytics")
st.markdown('<p class="subtitle">Explore restaurant trends, customer engagement and machine-learning clusters.</p>', unsafe_allow_html=True)

# Shared filters
with st.sidebar.expander("Global filters", expanded=True):
    cuisine_values = sorted(set(
        c.strip() for value in restaurant_df["Cuisines"].dropna()
        for c in str(value).split(",") if c.strip()
    ))
    chosen_cuisines = st.multiselect("Cuisine", cuisine_values, default=[])
    min_cost = int(max(0, restaurant_df["Cost"].min(skipna=True) or 0))
    max_cost = int(restaurant_df["Cost"].max(skipna=True) or 5000)
    if max_cost <= min_cost:
        max_cost = min_cost + 1
    cost_range = st.slider("Cost for two (₹)", min_cost, max_cost, (min_cost, max_cost))
    rating_range = st.slider("Average rating", 0.0, 5.0, (0.0, 5.0), 0.1)

filtered = restaurant_df.copy()
filtered = filtered[filtered["Cost"].between(cost_range[0], cost_range[1]) | filtered["Cost"].isna()]
filtered = filtered[filtered["Average_Rating"].between(rating_range[0], rating_range[1]) | filtered["Average_Rating"].isna()]
if chosen_cuisines:
    filtered = filtered[filtered["Cuisines"].fillna("").apply(
        lambda value: any(c in [x.strip() for x in str(value).split(",")] for c in chosen_cuisines)
    )]

if page == "Overview":
    st.subheader("Project overview")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Restaurants", f"{len(filtered):,}")
    c2.metric("Average rating", f"{filtered['Average_Rating'].mean():.2f}" if filtered["Average_Rating"].notna().any() else "N/A")
    c3.metric("Average cost", f"₹{filtered['Cost'].mean():,.0f}" if filtered["Cost"].notna().any() else "N/A")
    c4.metric("Customer reviews", f"{int(filtered['Review_Count'].sum()):,}")
    left,right = st.columns(2)
    with left:
        fig = px.histogram(filtered.dropna(subset=["Cost"]), x="Cost", nbins=18, title="Cost distribution")
        st.plotly_chart(style_chart(fig), use_container_width=True)
    with right:
        fig = px.histogram(reviews_clean.dropna(subset=["Rating"]), x="Rating", title="Customer rating distribution")
        st.plotly_chart(style_chart(fig), use_container_width=True)
    st.subheader("Restaurant sample")
    show_cols = ["Name","Cost","Cuisines","Average_Rating","Review_Count","KMeans_Cluster"]
    st.dataframe(filtered[show_cols].sort_values("Review_Count", ascending=False), use_container_width=True, hide_index=True)

elif page == "EDA Chart Explorer":
    st.subheader("Explore the 15 project visualizations")
    chart_options = [
        "1. Restaurant cost distribution",
        "2. Customer rating distribution",
        "3. Top cuisines",
        "4. Cost vs average rating",
        "5. Cuisine count vs average rating",
        "6. Customer rating boxplot",
        "7. Cost across top 5 cuisines",
        "8. Reviews per restaurant",
        "9. Average pictures vs review count",
        "10. Average rating by price category",
        "11. Top 10 restaurants by review count",
        "12. Rating vs pictures",
        "13. Average rating vs review count",
        "14. Correlation heatmap",
        "15. Pair plot"
    ]
    selected_chart = st.selectbox("Choose a chart", chart_options)
    fig = make_chart(selected_chart, filtered, reviews_clean, "KMeans_Cluster")
    if fig is not None:
        st.plotly_chart(style_chart(fig), use_container_width=True)
        st.caption("The chart responds to the global cuisine, cost and rating filters where applicable.")
    else:
        st.warning("Not enough data is available for this chart with the current filters.")

elif page == "Machine Learning":
    st.subheader("Unsupervised machine learning")
    algorithm = st.selectbox("Choose clustering algorithm", ["K-Means", "Agglomerative", "DBSCAN"])
    cluster_col = {"K-Means":"KMeans_Cluster","Agglomerative":"Agglomerative_Cluster","DBSCAN":"DBSCAN_Cluster"}[algorithm]
    sil, dbi, chi = metric_values(scaled_df, restaurant_df[cluster_col], ignore_noise=(algorithm=="DBSCAN"))
    a,b,c = st.columns(3)
    a.metric("Silhouette Score", f"{sil:.3f}" if pd.notna(sil) else "N/A")
    b.metric("Davies-Bouldin Index", f"{dbi:.3f}" if pd.notna(dbi) else "N/A")
    c.metric("Calinski-Harabasz Score", f"{chi:.2f}" if pd.notna(chi) else "N/A")
    st.caption("For DBSCAN, quality metrics are calculated after excluding noise points (-1), as in the notebook's evaluation approach.")
    chart_pick = st.selectbox("Cluster visualization", ["Cluster distribution","Cluster feature profile","Restaurant clusters (2D)"])
    if chart_pick == "Restaurant clusters (2D)":
        pca = None
        from sklearn.decomposition import PCA
        coords = PCA(n_components=2).fit_transform(scaled_df)
        plot_df = restaurant_df[["Name",cluster_col]].copy()
        plot_df["Component 1"] = coords[:,0]
        plot_df["Component 2"] = coords[:,1]
        fig = px.scatter(plot_df, x="Component 1", y="Component 2", color=cluster_col.astype(str) if False else plot_df[cluster_col].astype(str),
                         hover_name="Name", title=f"{algorithm} clusters (PCA projection)")
        st.plotly_chart(style_chart(fig), use_container_width=True)
    else:
        fig = make_chart(chart_pick, restaurant_df, reviews_clean, cluster_col)
        st.plotly_chart(style_chart(fig), use_container_width=True)
    st.markdown("**Cluster interpretation**")
    st.write("Compare the average cost, rating, review count, cuisine count and picture activity for each group. Cluster numbers are identifiers, not rankings.")
    st.dataframe(restaurant_df.groupby(cluster_col)[["Cost","Average_Rating","Review_Count","Cuisine_Count","Average_Pictures"]].mean().round(2), use_container_width=True)

elif page == "Restaurant Explorer":
    st.subheader("Find and compare restaurants")
    search = st.text_input("Search restaurant name")
    explorer = filtered.copy()
    if search:
        explorer = explorer[explorer["Name"].str.contains(search, case=False, na=False)]
    cluster_filter = st.selectbox("K-Means cluster", ["All"] + sorted(explorer["KMeans_Cluster"].unique().tolist()))
    if cluster_filter != "All":
        explorer = explorer[explorer["KMeans_Cluster"] == cluster_filter]
    columns = ["Name","Cost","Cuisines","Collections","Timings","Average_Rating","Review_Count","Average_Pictures","KMeans_Cluster"]
    st.dataframe(explorer[columns].sort_values("Average_Rating", ascending=False), use_container_width=True, hide_index=True)
    if len(explorer):
        chosen = st.selectbox("View restaurant details", explorer["Name"].tolist())
        row = explorer[explorer["Name"] == chosen].iloc[0]
        st.markdown(f"### {row['Name']}")
        st.write(f"**Cost for two:** ₹{row['Cost'] if pd.notna(row['Cost']) else 'Not available'}")
        st.write(f"**Cuisines:** {row['Cuisines']}")
        st.write(f"**Average rating:** {row['Average_Rating']:.2f}" if pd.notna(row["Average_Rating"]) else "**Average rating:** Not available")
        st.write(f"**Reviews counted:** {int(row['Review_Count'])}")
        if "Links" in row and pd.notna(row["Links"]):
            st.markdown(f"[Open restaurant page]({row['Links']})")

elif page == "Cluster Prediction":
    st.subheader("Predict a new restaurant's K-Means cluster")
    st.info("Enter the same five features used by your notebook. Cost and review count are log-transformed and all features are scaled before prediction.")
    with st.form("prediction_form"):
        col1,col2 = st.columns(2)
        with col1:
            cost = st.number_input("Cost for two (₹)", min_value=0.0, value=800.0, step=50.0)
            rating = st.number_input("Average rating", min_value=0.0, max_value=5.0, value=4.2, step=0.1)
            review_count = st.number_input("Review count", min_value=0, value=150, step=10)
        with col2:
            cuisine_count = st.number_input("Number of cuisines", min_value=0, value=3, step=1)
            average_pictures = st.number_input("Average pictures per review", min_value=0.0, value=5.0, step=0.5)
        submitted = st.form_submit_button("Predict cluster", type="primary")
    if submitted:
        new_row = pd.DataFrame([{
            "Cost": cost,
            "Average_Rating": rating,
            "Review_Count": review_count,
            "Cuisine_Count": cuisine_count,
            "Average_Pictures": average_pictures
        }])
        new_row["Cost"] = np.log1p(new_row["Cost"].clip(lower=0))
        new_row["Review_Count"] = np.log1p(new_row["Review_Count"].clip(lower=0))
        new_scaled = scaler.transform(new_row[model_df.columns])
        prediction = int(kmeans_model.predict(new_scaled)[0])
        st.success(f"This restaurant is assigned to K-Means Cluster {prediction}.")
        st.caption("Cluster IDs are labels only. Use the cluster profile on the Machine Learning page to understand the segment.")

st.sidebar.divider()
st.sidebar.caption("Built from the Zomato restaurant metadata and reviews project.")
