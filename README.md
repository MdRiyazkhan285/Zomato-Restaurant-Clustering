# Zomato Interactive Analytics Dashboard

## 1. Project structure

```text
Zomato_Interactive_Dashboard/
├── app.py
├── requirements.txt
├── README.md
└── data/
    ├── Zomato Restaurant names and Metadata.csv
    └── Zomato Restaurant reviews.csv
```

## 2. Run locally

Install Python 3.10 or newer, then open a terminal in this folder.

```bash
python -m venv .venv
```

Activate the environment:

**Windows**
```bash
.venv\Scripts\activate
```

**macOS/Linux**
```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the dashboard:

```bash
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`. Open it in your browser.

## 3. Dataset handling

The two original project CSV files are included in `data/`. You can also upload replacement CSVs from the dashboard sidebar.

## 4. Model notes

- The K-Means model follows the notebook's feature order:
  `Cost`, `Average_Rating`, `Review_Count`, `Cuisine_Count`, `Average_Pictures`.
- Missing numerical features are filled using column medians.
- `Cost` and `Review_Count` are transformed using `log1p`.
- The five features are standardized using `StandardScaler`.
- K-Means uses the best cluster count selected by Silhouette Score over k=2 through k=10 (bounded by available rows), matching the notebook's baseline model logic.
- Agglomerative uses Ward linkage and selects k by Silhouette Score.
- DBSCAN uses `eps=0.8` and `min_samples=4`, matching the notebook's baseline DBSCAN configuration.
- Standard scikit-learn Agglomerative and DBSCAN do not provide a normal `.predict()` method for unseen records. Therefore, the new-record prediction page uses K-Means only.

## 5. Important interpretation note

Cluster IDs such as 0, 1, and 2 are arbitrary group labels. They do not indicate a quality ranking. Use cluster profiles to interpret the groups.
