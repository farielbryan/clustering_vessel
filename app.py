import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from shapely.geometry import MultiPoint, Polygon
import folium
from streamlit_folium import st_folium
import os

# --- Configuration de la page ---
st.set_page_config(page_title="Geospatial Clustering", layout="wide")
st.title("📍 Analyse de Clusters Géospatiaux (DBSCAN)")

# --- Sidebar : Paramètres ---
st.sidebar.header("Configuration")

# --- MODIFICATION ICI : Entrée en KM et conversion en Degrés ---
# On propose un rayon entre 0.1 km (100m) et 10 km
eps_km = st.sidebar.slider("Rayon (EPS) en km", 0.1, 0.3, 0.1, step=0.1)
# Conversion pour l'algorithme (1 degré latitude ~= 111.32 km)
eps = eps_km / 111.32 

min_samples = st.sidebar.number_input("Min points par cluster", min_value=1, value=20)
sample_limit = st.sidebar.slider("Limite d'échantillonnage", 1000, 100000, 45000)

# --- Fonctions avec MISE EN CACHE (Optimisées) ---

@st.cache_data
def load_data(file_path_or_buffer, file_type):
    """Charge les données et les met en cache."""
    try:
        if file_type == 'csv':
            # Si c'est un fichier uploadé (buffer), on remet le curseur au début
            if hasattr(file_path_or_buffer, 'seek'):
                file_path_or_buffer.seek(0)
                # Lecture rapide pour détecter le séparateur
                content = file_path_or_buffer.getvalue().decode("utf-8").splitlines()
                sep = ',' if content[0].count(',') >= content[0].count(';') else ';'
                file_path_or_buffer.seek(0)
                return pd.read_csv(file_path_or_buffer, sep=sep)
            else:
                # Lecture fichier local (string path)
                with open(file_path_or_buffer, 'r', encoding='utf-8') as f:
                    first_line = f.readline()
                    sep = ',' if first_line.count(',') >= first_line.count(';') else ';'
                return pd.read_csv(file_path_or_buffer, sep=sep)
        else:
            return pd.read_excel(file_path_or_buffer)
    except Exception as e:
        return None

@st.cache_data
def detect_lon_lat(columns):
    """Détecte les noms des colonnes (nécessite une liste en entrée)."""
    cols_lower = {c.lower(): c for c in columns}
    lon_keys = ["longitude", "lon", "long", "lng", "x"]
    lat_keys = ["latitude", "lat", "y"]
    lon_col = next((cols_lower[k] for k in lon_keys if k in cols_lower), None)
    lat_col = next((cols_lower[k] for k in lat_keys if k in cols_lower), None)
    return lon_col, lat_col

@st.cache_data
def process_coords(df, lon_col, lat_col, limit):
    """Nettoie et échantillonne les données."""
    if len(df) > limit:
        df = df.sample(n=limit, random_state=42)
    
    coords = df[[lon_col, lat_col]].copy()
    coords.columns = ["lon", "lat"]
    
    # Conversion robuste en numérique
    for col in ["lon", "lat"]:
        if coords[col].dtype == object:
            coords[col] = pd.to_numeric(coords[col].astype(str).str.replace(',', '.'), errors="coerce")
            
    coords = coords.dropna().reset_index(drop=True)
    coords = coords[coords["lon"].between(-180, 180) & coords["lat"].between(-90, 90)]
    return coords

@st.cache_data
def run_dbscan(coords, eps, min_samples):
    """Exécute DBSCAN. Le cache évite de recalculer si les sliders ne bougent pas."""
    if len(coords) == 0: return coords, {}, 0, 0
    
    # Algorithme DBSCAN
    # Note: eps est ici reçu en degrés (après conversion)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords[["lon", "lat"]].values)
    
    coords_result = coords.copy()
    coords_result["cluster"] = db.labels_
    
    # Calcul des polygones (Convex Hulls)
    polygons = {}
    unique_labels = set(db.labels_)
    if -1 in unique_labels: unique_labels.remove(-1)
    
    for cid in sorted(unique_labels):
        pts = coords_result[coords_result["cluster"] == cid][["lon", "lat"]].values
        if len(pts) > 2:
            hull = MultiPoint(pts).convex_hull
            polygons[cid] = hull
            
    n_clusters = len(unique_labels)
    n_noise = list(db.labels_).count(-1)
    
    return coords_result, polygons, n_clusters, n_noise

# --- Logique Principale ---

DATA_DIR = "data"
# Liste des fichiers disponibles localement
available_ports = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")] if os.path.exists(DATA_DIR) else []

st.sidebar.header("Source des données")
source_mode = st.sidebar.radio("Choisir la source :", ["Fichier pré-chargé", "Uploader un fichier"])

df = None
file_name = "resultats"

# 1. Chargement des données
if source_mode == "Fichier pré-chargé":
    if available_ports:
        selected_file = st.sidebar.selectbox("Sélectionnez un port :", available_ports)
        file_path = os.path.join(DATA_DIR, selected_file)
        df = load_data(file_path, 'csv')
        file_name = os.path.splitext(selected_file)[0]
    else:
        st.warning("Aucun fichier CSV trouvé dans le dossier 'data/'.")
else:
    uploaded_file = st.file_uploader("Glissez votre fichier ici", type=["csv", "xlsx"])
    if uploaded_file:
        file_name = os.path.splitext(uploaded_file.name)[0]
        ftype = 'csv' if uploaded_file.name.endswith('.csv') else 'xlsx'
        df = load_data(uploaded_file, ftype)

# 2. Traitement et Affichage
if df is not None:
    # --- CORRECTION : .tolist() pour le cache ---
    lon_col, lat_col = detect_lon_lat(df.columns.tolist())
    
    if lon_col and lat_col:
        # Nettoyage
        coords = process_coords(df, lon_col, lat_col, sample_limit)
        
        # Clustering avec Spinner
        with st.spinner('Calcul des clusters en cours...'):
            # On passe 'eps' qui est maintenant converti en degrés
            coords_result, polygons, n_clusters, n_noise = run_dbscan(coords, eps, min_samples)

        # Métriques
        col1, col2, col3 = st.columns(3)
        col1.metric("Points total", len(coords_result))
        col2.metric("Clusters détectés", n_clusters)
        col3.metric("Bruit (Points isolés)", n_noise)

        # Carte Folium
        center = [coords_result["lat"].mean(), coords_result["lon"].mean()]
        m = folium.Map(location=center, zoom_start=11, tiles="Cartodb Positron")

        # Polygones des clusters
        for cid, poly in polygons.items():
            if isinstance(poly, Polygon):
                locations = [(y, x) for x, y in poly.exterior.coords]
                folium.Polygon(
                    locations, 
                    color="orange", 
                    weight=2,
                    fill=True, 
                    fill_opacity=0.4,
                    tooltip=f"Cluster {cid}"
                ).add_to(m)

        # Points (Échantillon léger pour ne pas surcharger le navigateur)
        viz_sample = coords_result.sample(min(len(coords_result), 1000))
        for _, row in viz_sample.iterrows():
            c = "#ff0000" if row['cluster'] == -1 else "#3388ff"
            folium.CircleMarker(
                [row['lat'], row['lon']], 
                radius=1.5, 
                color=c, 
                fill=True,
                fill_opacity=0.8
            ).add_to(m)

        # Affichage de la carte
        st_folium(m, width=None, height=500, returned_objects=[])

        # Bouton Télécharger
        csv = coords_result.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Télécharger les résultats (CSV)",
            data=csv,
            file_name=f"{file_name}_clustered.csv",
            mime="text/csv",
        )

    else:
        st.error(f"Colonnes de coordonnées non trouvées. Colonnes disponibles : {list(df.columns)}")