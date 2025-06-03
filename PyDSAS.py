import numpy as np
import pandas as pd
import geopandas as gpd
import contextily as ctx
import pymannkendall as mk
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

from shapely.geometry import LineString, Polygon, MultiPolygon, MultiLineString, Point, GeometryCollection
from sklearn.linear_model import RANSACRegressor, LinearRegression
from shapely.ops import unary_union, linemerge, polygonize, snap
from sklearn.metrics import cohen_kappa_score
from contextlib import redirect_stdout
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
from dask.distributed import Client
from datetime import datetime
from scipy.stats import t
from tqdm import tqdm

from creation_baseline_function import recursiveFileSearch, createBasicFolderStructure, findIntoFolderStructure, extract_date_from_filename, classer_groupe, show_progress, extract_lines, extract_exterior_from_lines, flatten_geometries, azimut, extract_open_line_from_closed, create_transects,calc_SCE, calc_NSM, calc_EPR, calc_LRR, calc_WLR, calc_LMS,couleur_evolution, calcul_et_afficher_statistiques_annuelles, calcul_moyennes_par_groupe

##################################################################################

# Récupérer le dossier racine du projet (celui où se trouve ce script)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Chemin vers le dossier contenant les traits de côtes (input_data/sds)
dossier_base = os.path.join(BASE_DIR, "input_data", "sds")

print(f"dossier_base : {dossier_base}")

# Chemin vers la couche beaches.shp dans aux_data
chemin_terre = os.path.join(BASE_DIR, "aux_data", "beaches.shp")


print(f"chemin_terre : {chemin_terre}")

# Chemin pour enregistrer les résultats dans output_data
output_dir = os.path.join(BASE_DIR, "output_data")

print(f"output_dir : {output_dir}")

# Création du dossier output_data s'il n'existe pas
os.makedirs(output_dir, exist_ok=True)

# Construire le chemin vers le fichier Excel dans input_data
maree_excel_path = os.path.join(BASE_DIR, "input_data", "Maree_modifie.xlsx")

# Vérifier que le fichier existe
if not os.path.isfile(maree_excel_path):
    raise FileNotFoundError(f"Le fichier des marées est introuvable : {maree_excel_path}. "
                            "Veuillez le placer dans le dossier input_data.")

# Charger le fichier Excel
maree_df = pd.read_excel(maree_excel_path)

print(f"Données de marée chargées depuis : {maree_excel_path}")

# Chemin vers le dossier input_data où sont stockés les fichiers CSV
chemin_candhis = os.path.join(BASE_DIR, "input_data")

# Liste des fichiers CSV
fichiers_candhis = [
    'Candhis_97403_2016_arch.csv',
    'Candhis_97403_2017_arch.csv',
    'Candhis_97403_2018_arch.csv',
    'Candhis_97403_2019_arch.csv',
    'Candhis_97403_2020_arch.csv',
    'Candhis_97403_2021_arch.csv',
    'Candhis_97403_2022_arch.csv',
    'Candhis_97403_2023_arch.csv',
    'Candhis_97403_2024_arch.csv'
]

# Construction des chemins complets vers chaque fichier CSV
chemins_fichiers_candhis = [os.path.join(chemin_candhis, f) for f in fichiers_candhis]

# Vérification optionnelle que tous les fichiers existent
for chemin_fichier in chemins_fichiers_candhis:
    if not os.path.isfile(chemin_fichier):
        print(f"Attention : fichier introuvable {chemin_fichier}")

# Exemple d'utilisation : affichage des chemins
for chemin_fichier in chemins_fichiers_candhis:
    print(f"Fichier CSV prêt à être utilisé : {chemin_fichier}")

total_steps = 10
current_step = 1

show_progress(current_step, total_steps, "Chercher les traits de côtes et de créer des attribut OBJECTED, SHAPE, SHAPE_length, DATE, UNCERTAINTY, Tidal_height et Tidal_condition")

all_gdfs = []

for fichier in os.listdir(dossier_base):
    if fichier.endswith('_lines.shp'):
        chemin_shp = os.path.join(dossier_base, fichier)
        gdf = gpd.read_file(chemin_shp)

        if len(all_gdfs) == 0:
            print("ERREUR : Aucun fichier shapefile n'a été trouvé ou chargé.")

        gdf = gdf.to_crs(epsg=epsg_reunion)

        # ------------------ Extraire la date depuis le nom du fichier ------------------

        date_extrait = extract_date_from_filename(fichier)
        date_extrait_dt = pd.to_datetime(date_extrait, format='%Y%m%d', errors='coerce')

        # ------------------ Créer OBJECTID unique localement (par entité) ------------------

        if 'OBJECTED' not in gdf.columns:
            gdf['OBJECTED'] = range(1, len(gdf) + 1)

        # shape : type géométrique

        gdf['SHAPE'] = gdf.geometry.geom_type

        # shape_length : longueur de chaque géométrie

        gdf['SHAPE_length'] = gdf.geometry.length

        # Ajouter ou remplir la colonne date avec la date extraite

        if 'DATE' not in gdf.columns:
            gdf['DATE'] = date_extrait_dt
        else:
            gdf['DATE'] = gdf['DATE'].fillna(date_extrait_dt)

        # uncertainty : si absent, mettre NA

        if 'UNCERTAINTY' not in gdf.columns:
            gdf['UNCERTAINTY'] = pd.NA

        all_gdfs.append(gdf)

print(f"Nombre total de GeoDataFrames collectés : {len(all_gdfs)}")

if len(all_gdfs) == 0:
    print("Aucun trait de côte trouvé, vérifiez vos dossiers et le filtre de nom de fichier.")
    exit(1)

current_step += 1

print()
print()

show_progress(current_step, total_steps, "Début de la fusion des GeoDataFrames...")

# --- 1. Concaténer tous les GeoDataFrames en un seul GeoDataFrame fusionné ---
chunks = []
for gdf in all_gdfs:
    chunks.append(gdf)

gdf_fusion = gpd.GeoDataFrame(pd.concat(chunks, ignore_index=True), crs=f"EPSG:{epsg_reunion}")

# --- 2. Conversion des colonnes de date en datetime pour assurer la correspondance ---
#maree_df['Date_tide'] = maree_df['Date_tide'].astype(str)
maree_df['Date_tide'] = maree_df['Date_tide'].astype(str).str.split('.').str[0].str.strip()

maree_df['Date_tide'] = pd.to_datetime(maree_df['Date_tide'], format='%Y%m%d', errors='coerce')

# S'assurer que Tidal_height est bien un float
maree_df['Tidal_height'] = maree_df['Tidal_height'].astype(float)

#print(maree_df['Date_tide'].head())

# S'assurer que Tidal_condition est bien une chaîne
maree_df['Tidal_condition'] = maree_df['Tidal_condition'].astype(str).str.strip()

# 2. Conversion de la colonne DATE du GeoDataFrame
gdf_fusion['DATE'] = pd.to_datetime(gdf_fusion['DATE'], errors='coerce')

# Vérification
#print(gdf_fusion['DATE'].head())

# --- 3. Fusionner (merge) gdf_fusion avec maree_df sur les colonnes de date ---
gdf_fusion = gdf_fusion.merge(
    maree_df[['Date_tide', 'Tidal_height', 'Tidal_condition']],
    how='left',                # jointure gauche pour garder toutes les lignes de gdf_fusion
    left_on='DATE',            # colonne dans gdf_fusion
    right_on='Date_tide'       # colonne dans maree_df
)

# Remplissage des valeurs manquantes
gdf_fusion['Tidal_height'] = gdf_fusion['Tidal_height'].fillna(0)
gdf_fusion['Tidal_condition'] = gdf_fusion['Tidal_condition'].fillna('NAN')

# --- 5. Supprimer la colonne 'Date_tide' issue de la jointure si elle n'est plus nécessaire ---
gdf_fusion.drop(columns=['Date_tide'], inplace=True)

# Etat des traits de côte (TDC)

gdf_fusion['annee'] = gdf_fusion['DATE'].dt.year

# Définition des périodes
periode_couleurs = {
    'TDC ancien': ('Blues', range(2016, 2019)),              # 2016, 2017, 2018
    'TDC intermédiaire': ('Oranges', range(2019, 2022)),     # 2019, 2020, 2021
    'TDC récent': ('Greens', range(2022, 2026)),             # 2022, 2023, 2024, 2025
}

def attribuer_etat(date):
    if pd.isnull(date):
        return 'NA'
    annee = date.year
    if 2016 <= annee < 2019:
        return 'TDC ancien'
    elif 2019 <= annee < 2022:
        return 'TDC intermédiaire'
    elif 2022 <= annee <= 2025:
        return 'TDC récent'
    else:
        return 'NA'

def couleur_etat_ligne(row):
    etat = row['etat']
    annee = row['annee']
    if pd.isnull(annee) or etat == 'NA':
        return '#cccccc'
    if etat in periode_couleurs:
        palette_name, annees = periode_couleurs[etat]
        annees = list(annees)
        norm = mcolors.Normalize(vmin=min(annees), vmax=max(annees))
        cmap = plt.get_cmap(palette_name)
        # Si l'année dépasse la borne, on prend le max
        if annee < min(annees):
            annee = min(annees)
        if annee > max(annees):
            annee = max(annees)
        couleur = mcolors.to_hex(cmap(norm(annee)))
        return couleur
    else:
        return '#cccccc'


gdf_fusion['etat'] = gdf_fusion['DATE'].apply(attribuer_etat)
gdf_fusion['couleur_etat'] = gdf_fusion.apply(couleur_etat_ligne, axis=1)

# Calcul de l'incertitude totale (UNCERTAINTY) pour DSAS/WLR

# Paramètres recommandés pour Sentinel-2: Réf: Liu, X., Lu, Z., Yang, W., Huang, M., & Tong, X. (2018). Dynamic monitoring and vibration analysis of ancient bridges by ground-based microwave interferometry and the ESMD method. Remote Sensing, 10(5), 770   || Ahmed, M., Sultan, M., Elbayoumi, T., & Tissot, P. (2019). Forecasting GRACE data over the African watersheds using artificial neural networks. Remote Sensing, 11(15), 1769.
incert_geo = 10.0   # géoréférencement (m) || Réf: https://sentiwiki.copernicus.eu/web/document-library#DocumentLibrary-SENTINEL-2DocumentsLibrary-S2-Documents
incert_digit = 10.0 # digitalisation (m)
incert_maree = 0.010 # marée (m), valeur recommandée

# Ajout de la colonne UNCERTAINTY
gdf_fusion['UNCERTAINTY'] = (incert_maree**2 + incert_geo**2 + incert_digit**2)**0.5

# Listes des BEACH_CODE par type de plage

plages_sable_blanc_sans_recif = [8193, 8212, 8213, 8214, 8281]
plages_sable_blanc_avec_recif = [8192, 8194, 8195, 8196, 8197, 8198, 8199, 8200, 8201, 8202, 8203, 8204, 8205, 8206, 8207, 8208, 8209, 8210, 8211, 8215, 8216, 8217, 8221, 8222, 8233, 8234, 8280]
plages_sable_noir_sans_recif = [8224, 8267, 8271, 8272, 8277, 8278, 8279]
plages_sable_noir_avec_recif = [8225, 8226]
plages_mixtes_sans_recif = [8232, 8268, 8269, 8270, 8273, 8275, 8276, 8283, 8218, 8219, 8220, 8227, 8193]
plages_galets = [8228, 8230, 8231, 8235, 8237, 8238, 8239, 8240, 8241, 8242, 8243, 8244, 8245, 8246, 8247, 8248, 8249, 8250, 8251, 8253, 8254, 8255, 8256, 8257, 8258, 8259, 8260, 8262, 8263, 8265]
embouchures = [8223, 8229, 8236, 8252, 8261, 8264, 8266, 8274, 8282]

# ------------------ Group : [long integer] défault 0 ------------------

beach_codes = gdf_fusion.get('BEACH_CODE')
if beach_codes is not None:
    gdf_fusion['GROUP'] = beach_codes.apply(classer_groupe)
else:
    print("La colonne 'BEACH_CODE' est absente.")
    gdf_fusion['GROUP'] = 0

# Chargement et fusion des données CANDHIS

dfs = []
for fichier in fichiers_candhis:
    chemin = os.path.join(chemin_candhis, fichier)
    try:
        df = pd.read_csv(
            chemin,
            sep=';',
            usecols=['DateHeure', 'HM0'],
            parse_dates=['DateHeure']
        )
        dfs.append(df)
    except FileNotFoundError:
        print(f"Fichier non trouvé : {chemin}")
    except Exception as e:
        print(f"Erreur lors du chargement de {fichier} : {str(e)}")

if not dfs:
    raise ValueError("Aucun fichier CANDHIS chargé correctement.")

# Fusionner tous les DataFrames en un seul
candhis_fusion = pd.concat(dfs, ignore_index=True)

#print(candhis_fusion.head())

# Nettoyage et agrégation par date (moyenne journalière)
candhis_fusion['Date'] = candhis_fusion['DateHeure'].dt.normalize()  # Enlève l'heure
candhis_par_date = candhis_fusion.groupby('Date')['HM0'].mean().reset_index()

# Chemin complet du fichier CSV à sauvegarder
chemin_sortie = os.path.join(output_dir, "candhis_fusion.csv")

# Sauvegarde du DataFrame pandas sans l'index
candhis_par_date.to_csv(chemin_sortie, index=False, encoding='utf-8')

# ------------------ Étape 2 : Jointure avec le GeoDataFrame ------------------
# Conversion de la colonne DATE en type date
gdf_fusion['DATE'] = pd.to_datetime(gdf_fusion['DATE']).dt.normalize()

# Jointure des données HMO
gdf_fusion = gdf_fusion.merge(
    candhis_par_date,
    how='left',
    left_on='DATE',
    right_on='Date'
)

# Remplissage des valeurs manquantes
gdf_fusion['HM0'] = gdf_fusion['HM0'].fillna(0)

# Nettoyage final
gdf_fusion.drop(columns=['Date'], inplace=True, errors='ignore')

# Chemin complet du fichier shapefile à sauvegarder
output_path_fusion = os.path.join(output_dir, "fusion_trait_de_cote.shp")

# Sauvegarde du GeoDataFrame dans un shapefile
gdf_fusion.to_file(output_path_fusion)

print(f"GeoDataFrame fusionné sauvegardé dans : {output_path_fusion}")

current_step += 1

print()
print()
# ------------------ Création des buffers individuels de 20 m autour de chaque trait de côte ------------------

show_progress(current_step, total_steps, "Création des buffers individuels...")

buffer_distance = 20  # ------------------ en mètres ------------------

#print(f"Création des buffers individuels de {buffer_distance} m...")
logging.info(f'Création des buffers individuels de {buffer_distance} m.')
buffers = gdf_fusion.geometry.buffer(buffer_distance)

# ------------------ Fusionner tous les buffers en un seul polygone ------------------

#print("Fusion des buffers en un seul polygone...")
logging.info('Fusion des buffers en un seul polygone...')
buffer_union = unary_union(buffers)

# ------------------ Créer un GeoDataFrame à partir du polygone fusionné ------------------

buffer_union_gdf = gpd.GeoDataFrame(geometry=[buffer_union], crs=gdf_fusion.crs)

# ------------------ Sauvegarder ce buffer fusionné pour contrôle ------------------

output_path = os.path.join(output_dir, "buffer_union.shp")

# Sauvegarde du GeoDataFrame dans un shapefile
buffer_union_gdf.to_file(buffer_union_path)

print(f"GeoDataFrame fusionné sauvegardé dans : {output_path}")

current_step += 1

print()
print()

# ------------------ Extraction de la baseline : fusion des buffer et extraction suivant la couche beaches.shp ------------------

show_progress(current_step, total_steps, "Fusion des buffers...")

baseline_geom = buffer_union_gdf.geometry.intersection(terre.geometry.union_all())

# ------------------ Filtrer les géométries non vides ------------------

baseline_geoms = [geom for geom in baseline_geom if not geom.is_empty]
#print(f"Nombre de géométries non vides dans la baseline : {len(baseline_geoms)}")

if len(baseline_geoms) == 0:
    print("Attention : aucune géométrie issue de l'intersection. Vérifiez les données et le buffer.")

# ------------------ Créer GeoDataFrame baseline------------------

baseline = gpd.GeoDataFrame(geometry=baseline_geoms, crs=gdf_fusion.crs)

# ------------------ Sauvegarder la baseline finale ------------------

output_path = os.path.join(output_dir, "baseline_final.shp")

# Sauvegarde du GeoDataFrame dans un shapefile
baseline.to_file(output_path)

print(f"GeoDataFrame fusionné sauvegardé dans : {output_path}")

current_step += 1

print()
print()

# ------------------ Extraction de la frontière (boundary) du polygone intersecté ------------------

show_progress(current_step, total_steps, "Extraction de la frontière du polygone intersecté (baseline)...")

boundaries = baseline.boundary

# ------------------ Filtrer pour ne garder que les lignes qui touchent la plage (évite les contours côté mer) ------------------

#print("Filtrage des lignes pour ne garder que la bordure en contact avec la plage...")
beach_union = terre.geometry.unary_union  # ------------------ ou .union_all() ------------------

baseline_lines = []
for line in boundaries:
    # ------------------ On garde uniquement les lignes qui touchent la plage ------------------
    if line.intersects(beach_union):
        baseline_lines.append(line)

#print(f"Nombre de segments de baseline extraits : {len(baseline_lines)}")

# ------------------ Création d'un GeoDataFrame pour la baseline linéaire------------------

baseline_line_gdf = gpd.GeoDataFrame(geometry=baseline_lines, crs=terre.crs)

# ------------------ Sauvegarde en shapefile ------------------

output_path = os.path.join(output_dir, "baseline_final_line.shp")

# Sauvegarde du GeoDataFrame dans un shapefile
baseline_line_gdf.to_file(output_path)

print(f"GeoDataFrame fusionné sauvegardé dans : {output_path}")

current_step += 1

print()
print()

# ------------------ Extraction de la frontière du polygone intersecté (baseline) ------------------
show_progress(current_step, total_steps, "Extraction de la frontière du polygone intersecté en contour (baseline)...")

all_lines = []
for geom in baseline.boundary:
    all_lines.extend(extract_lines(geom))

# ------------------ Fusionner tous les segments en une seule MultiLineString ------------------

merged = linemerge(all_lines)

# ------------------ S'assurer d'avoir une liste de LineString pour itérer sur chaque segment ------------------

if isinstance(merged, LineString):
    lines = [merged]
elif isinstance(merged, MultiLineString):
    lines = list(merged.geoms)
else:
    lines = []

# ------------------ Calculer la frontière de la plage ------------------

beach_boundary = terre.boundary.unary_union

# ------------------ Filtrer les segments proches de la plage ------------------

tolerance = 3.0  # ------------------ en mètres, à ajuster selon la précision de tes données ------------------
baseline_lines = []
for line in lines:
    if line.distance(beach_boundary) < tolerance:
        baseline_lines.append(line)

#print(f"Nombre de segments de baseline côté terre extraits : {len(baseline_lines)}")

# ------------------ Création d'un GeoDataFrame pour la baseline linéaire côté terre (sur l'estran ou dune) ------------------

baseline_line_gdf = gpd.GeoDataFrame(geometry=baseline_lines, crs=terre.crs)

# ------------------ Sauvegarde en shapefile ------------------

baseline_line_path = os.path.join(output_dir, "baseline_final_line_terre.shp")

# Sauvegarde du GeoDataFrame dans un shapefile

baseline_line_gdf.to_file(baseline_line_path)

print(f"Baseline linéaire côté terre sauvegardée : {baseline_line_path}")

current_step += 1

print()
print()

# ------------------ Fussion des bouts des contours de chaque polygones  en une bloc de polygone ------------------

show_progress(current_step, total_steps, "Extraire tous les LineString simple...")

gdf = baseline_line_gdf

all_lines = []
for geom in gdf.geometry:
    all_lines.extend(extract_lines(geom))

# ------------------ Fusionner les bouts de lignes proches (< 1 m) en utilisant un buffer puis boundary puis linemerge ------------------

buffered = [line.buffer(0.5) for line in all_lines]  # ------------------ 0.5 m de chaque côté = 1 m total ------------------
dissolved = unary_union(buffered)

# ------------------ Extraire la frontière du polygone fusionné ------------------

boundaries = []
if dissolved.geom_type == 'Polygon':
    boundaries = [dissolved.boundary]
elif dissolved.geom_type == 'MultiPolygon':
    for poly in dissolved.geoms:
        boundaries.append(poly.boundary)
elif dissolved.geom_type == 'GeometryCollection':
    for geom in dissolved.geoms:
        if geom.geom_type in ['Polygon', 'MultiPolygon']:
            boundaries.append(geom.boundary)
else:
    boundaries = []

# ------------------ Extraire tous les LineString simples des boundaries ------------------

final_lines = []
for geom in boundaries:
    final_lines.extend(extract_lines(geom))

# ------------------ Fusionner les segments en lignes continues ------------------

merged = linemerge(final_lines)
if isinstance(merged, LineString):
    merged_lines = [merged]
elif isinstance(merged, MultiLineString):
    merged_lines = list(merged.geoms)
else:
    merged_lines = []

# Création du GeoDataFrame fusionné en mémoire
gdf_merged = gpd.GeoDataFrame(geometry=merged_lines, crs=gdf.crs)

# Chemin de sortie shapefile fusionné
output_path_merged = os.path.join(output_dir, "baseline_final_line_terre_merged.shp")

# Sauvegarde
gdf_merged.to_file(output_path_merged)
print(f"Lignes fusionnées sauvegardées dans : {output_path_merged}")

current_step += 1

show_progress(current_step, total_steps, "Extraction de la la couche externe sans avoir compté les ligne interne qui sont les bouts des polygones auparavant...")

# ------------------ Extraction de la la couche externe sans avoir compté les ligne interne qui sont les bouts des polygones auparavant ------------------

gdf_exterior = extract_exterior_from_gdf(gdf_merged)
if gdf_exterior is not None:
    output_path_exterior = os.path.join(output_dir, "baseline_final_line_terre_exterior.shp")
    gdf_exterior.to_file(output_path_exterior)
    print(f"Bordure externe sauvegardée dans : {output_path_exterior}")

current_step += 1

print()
print()

show_progress(current_step, total_steps, "Extraction de la vraie baseline qui est la couche externe en contact de l'estran ou dune...")

# ------------------ Extraction de la vraie baseline qui est la couche externe en contact de l'estran ou dune à partir de la transformation de la couche externe précédent en vecteur polyligne grâce à une couche baseline_manual créer manuellement sur QGIS pour avoir automatiser l'acquisition du beseline ------------------

path_closed_lines = output_path_exterior
path_manual_lines = os.path.join(BASE_DIR, "input_data", "baseline_manual_2.shp")
output_path = os.path.join(output_dir, "baseline_final_line_terre_clipped.shp")

extract_open_line_from_closed(
    path_closed_lines=path_closed_lines,
    path_manual_lines=path_manual_lines,
    output_path=output_path,
    buffer_distance=10
)

current_step += 1

print()
print()

# ------------------ DEBUTE DE LA CALCUL DES STATISTIQUES AVEC DSAS ------------------
logging.info('DEBUTE DE LA CALCUL DES STATISTIQUES AVEC DSAS...')

show_progress(current_step, total_steps, "Création du TRANSECT...")

# ------------------ La couche de shoreline ------------------

chemin_trait = output_path_fusion

try:
    traits_gdf = gpd.read_file(chemin_trait, engine = 'fiona')
    print(f"Fichier chargé avec succès depuis : {traits_gdf}")
except ImportError:
    print("Erreur : le package 'fiona' est requis pour lire les shapefiles.")
except Exception as e:
    print(f"Erreur lors du chargement du shapefile : {e}")

# ------------------ La couche de baseline ------------------
chemin_baselines = os.path.join(output_dir, "baseline_final_line_terre_clipped.shp")

try:
    chemin_baseline = gpd.read_file(chemin_baselines, engine='fiona')
    print(f"Fichier chargé avec succès depuis : {chemin_baselines}")
except ImportError:
    print("Erreur : le package 'fiona' est requis pour lire les shapefiles.")
except Exception as e:
    print(f"Erreur lors du chargement du shapefile : {e}")

# ------------------ Vérifier que le fichier .shp existe ------------------

if not os.path.isfile(chemin_baselines):
    print(f"Fichier non trouvé : {chemin_baselines}")
else:
    print(f"Fichier trouvé : {chemin_baselines}")

# ------------------ Nombre de transects à calculer ------------------

N_TRANSECTS = 3002
TRANSECT_LENGTH = 250

# ------------------ Chargement de la baseline ------------------

print("Chargement de la baseline...")
baseline_gdf = gpd.read_file(chemin_baselines)
baseline_union = baseline_gdf.geometry.union_all()

# ------------------ Fusionner les segments en une seule ligne continue ------------------

merged = linemerge(baseline_union)

if not isinstance(merged, LineString):
    baseline = merged

elif isinstance(merged, MultiLineString):
    # Choisir la ligne la plus longueur
    baseline = max(merged.geoms, key=lambda l: l.length)

else:
    raise ValueError("La baseline ne peut pas être convertie en une LineString unique")

print(f"Baseline traitée : {type(baseline)}, longueur = {baseline.length}")

# ------------------ Créations des transects ------------------

print(f"Création de {N_TRANSECTS} transects sur la baseline...")

#print(f"Création de {N_TRANSECTS} transects sur la baseline...")
transects = create_transects(baseline, n=N_TRANSECTS, length=TRANSECT_LENGTH)
#print(f"{len(transects)} transects créés.")

# Création du GeoDataFrame
transect_gdf = gpd.GeoDataFrame(
    {'transect_id': range(len(transects))},
    geometry=transects,
    crs=baseline_gdf.crs
)

# ------------------ Enregistrement de la première couche de transect pour un test ------------------

output_path_tr_in = os.path.join(output_dir, "transects_initial.shp")

transect_gdf.to_file(output_path_tr_in)

# ------------------ On suppose que chaque entité a une colonne 'date' (au format YYYYMMDD) ------------------

dates = traits_gdf['DATE'].unique()
results = []

for date in sorted(dates):
    # ------------------ Sélectionner les traits de côte pour cette date ------------------

    gdf = traits_gdf[traits_gdf['DATE'] == date]
    trait = gdf.geometry.unary_union
    for idx, transect in enumerate(transects):
        inter = trait.intersection(transect)
        if inter.is_empty:
            dist = None
        else:
            # ------------------ Si multipoint, on prend le point le plus proche du centre du transect ------------------
            if inter.geom_type == 'MultiPoint':
                centre = transect.interpolate(0.5, normalized=True)
                distances_pts = [centre.distance(pt) for pt in inter.geoms]
                point_sel = inter.geoms[np.argmin(distances_pts)]
                dist = transect.project(point_sel)
            elif inter.geom_type == 'Point':
                dist = transect.project(inter)
            else:
                dist = None
        results.append({
            "DATE": date,
            "transect_id": idx,
            "distance_transect": dist
        })

df_distances = pd.DataFrame(results)

output_csv_path = os.path.join(output_dir, "distances_transects.csv")

# Sauvegarder le DataFrame en CSV sans l'index
df_distances.to_csv(output_csv_path, index=False, encoding='utf-8')

print(f"Distances des transects sauvegardées dans : {output_csv_path}")

urrent_step += 1

print()
print()

# ------------------ Calcul des modèles statistiques en DSAS ------------------
show_progress(current_step, total_steps, "Calcul des modèles statistiques en DSAS...")
# ------------------ SCE (Shoreline Change Enveloppe): distance measurement ------------------

# ------------------ Création du tableau pour stockés tous les modèles statistiques calculer ------------------

stats = []

print(df_distances.columns)

for transect_id, group in df_distances.groupby('transect_id'):
    group = group.dropna(subset=['distance_transect'])
    if group.empty:
        stats.append({'transect_id': transect_id, 'SCE': np.nan, 'NSM': np.nan, 'EPR': np.nan, 'LRR': np.nan, 'WLR': np.nan, 'LMS': np.nan})
        continue
    group = group.sort_values('DATE')
    dates_pd = pd.to_datetime(group['DATE'], errors='coerce', infer_datetime_format=True)
    distances = group['distance_transect']

    sce = calc_SCE(distances)
    nsm = calc_NSM(distances)
    years = (dates_pd.max() - dates_pd.min()).days / 365.25
    epr = calc_EPR(nsm, years)
    lrr = calc_LRR(dates_pd, distances)

    # ------------------ Pour WLR, on peut utiliser des poids si on a une colonne d'incertitude, sinon None (Incertitude vient de marée, géoréférecement des images satellitaire, ....) ------------------

    weights = None
    if 'uncertainty' in group.columns:
        weights = 1 / (group['UNCERTAINTY'] ** 2)  # Réf: https://pubs.usgs.gov/of/2018/1179/ofr20181179.pdf et Draper, N. R., & Smith, H. (1998). Applied regression analysis (Vol. 326). John Wiley & Sons.
    else:
        weights = None

    wlr = calc_WLR(dates_pd, distances, weights)
    lms = calc_LMS(dates_pd, distances)

    stats.append({'transect_id': transect_id, 'SCE': sce, 'NSM': nsm, 'EPR': epr, 'LRR': lrr, 'WLR': wlr, 'LMS': lms})

df_stats = pd.DataFrame(stats)

output_csv_path = os.path.join(output_dir, "stats_DSAS.csv")

# Sauvegarder le DataFrame en CSV sans l'index
df_stats.to_csv(output_csv_path, index=False, encoding='utf-8')

print(f"Distances des transects sauvegardées dans : {output_csv_path}")

# ------------------ Fusion des stats avec transects ------------------

transect_final = transect_gdf.merge(df_stats, on='transect_id', how='left')

# ------------------ Sauvegarder la couche finale ------------------

path_transects = os.path.join(output_dir, "transects_DSAS_final.shp")

# Sauvegarde du GeoDataFrame dans un shapefile

transect_final.to_file(path_transects)

print(f"Baseline linéaire côté terre sauvegardée : {path_transects}")


current_step += 1

print()
print()

# ------------------ Fusion de tous les couches de baseline, trait de côte et le transects qui est la résultats finale ------------------

show_progress(current_step, total_steps, "Couche TRANSECT finale fusion de tous les couches baseline et trait de côte...")

# ------------------ Chargement des couches ------------------

baseline_gdf = gpd.read_file(chemin_baselines)
fusion_trait_gdf = gpd.read_file(output_path_fusion)
transects_gdf = gpd.read_file(path_transects)

# ------------------ Harmoniser les CRS (important pour les jointures spatiales) ------------------

fusion_trait_gdf = fusion_trait_gdf.to_crs(baseline_gdf.crs)
#fusion_trait_gdf = fusion_trait_gdf.to_crs(by='DATE')
transects_gdf = transects_gdf.to_crs(baseline_gdf.crs)

# ------------------ Jointure spatiale transects + baseline (ex : intersect) ------------------

transects_baseline = gpd.sjoin(transects_gdf, baseline_gdf, how="left", predicate="intersects")

# ------------------ Nettoyage des colonnes problématiques------------------

for gdf in [transects_baseline, fusion_trait_gdf]:
    for col in ['index_left', 'index_right']:
        if col in gdf.columns:
            gdf.drop(columns=[col], inplace=True)
    gdf.reset_index(drop=True, inplace=True)

# ------------------ Jointure spatiale transects_baseline + fusion_trait_de_cote ------------------

transects_full = gpd.sjoin(transects_baseline, fusion_trait_gdf, how="left", predicate="intersects")

# ------------------ Nettoyer les colonnes issues des index des jointures ------------------

transects_full = transects_full.drop(columns=["index_right", "index_right_right"], errors='ignore')

# Ajouter l'attribut 'evolution

transects_full['Evolution_LRR [m/an]'] = transects_full['LRR'].apply(lambda x: classify_evolution(x, seuil=1e-4))
transects_full['Evolution_WLR [m/an]'] = transects_full['WLR'].apply(lambda x: classify_evolution(x, seuil=1e-4))

# ------------------ Sauvegarder la couche fusionnée finale ------------------

path_transects_fus = os.path.join(output_dir, "transects_fusionnees.shp")

# Sauvegarde du GeoDataFrame dans un shapefile

transects_full.to_file(path_transects)

print(f"Baseline linéaire côté terre sauvegardée : {path_transects}")

current_step += 1

print()
print()

show_progress(current_step, total_steps, "Création de la couche d'évolution ......")

# ------------------ Chargement des couches ------------------

baseline_gdf = gpd.read_file(chemin_baselines)
fusion_trait_gdf = gpd.read_file(output_path_fusion)
transects_gdf = gpd.read_file(path_transects)

# ------------------ Harmoniser les CRS (important pour les jointures spatiales) ------------------

fusion_trait_gdf = fusion_trait_gdf.to_crs(baseline_gdf.crs)
#fusion_trait_gdf = fusion_trait_gdf.to_crs(by='DATE')
transects_gdf = transects_gdf.to_crs(baseline_gdf.crs)

# ------------------ Jointure spatiale transects + baseline (ex : intersect) ------------------

transects_baseline = gpd.sjoin(transects_gdf, baseline_gdf, how="left", predicate="intersects")

# ------------------ Nettoyage des colonnes problématiques------------------

for gdf in [transects_baseline, fusion_trait_gdf]:
    for col in ['index_left', 'index_right']:
        if col in gdf.columns:
            gdf.drop(columns=[col], inplace=True)
    gdf.reset_index(drop=True, inplace=True)

# ------------------ Jointure spatiale transects_baseline + fusion_trait_de_cote ------------------

transects_full = gpd.sjoin(transects_baseline, fusion_trait_gdf, how="left", predicate="intersects")

# ------------------ Nettoyer les colonnes issues des index des jointures ------------------

transects_full = transects_full.drop(columns=["index_right"], errors='ignore')

# **CORRECTION 2: Ajouter les colonnes d'évolution manquantes**
if 'LRR' in transects_full.columns:
    transects_full['Evolution_LRR [m/an]'] = transects_full['LRR'].apply(lambda x: classify_evolution(x, seuil=1e-4))
else:
    print("Attention: La colonne 'LRR' n'existe pas dans les données")
    transects_full['Evolution_LRR [m/an]'] = 'NA'

if 'WLR' in transects_full.columns:
    transects_full['Evolution_WLR [m/an]'] = transects_full['WLR'].apply(lambda x: classify_evolution(x, seuil=1e-4))

# --- 1. Préparer les données géospatiales ---
# Filtrer la période et nettoyer les données
transects_full['year'] = pd.to_datetime(transects_full['DATE'], dayfirst=True, errors='coerce').dt.year
df_period = transects_full[(transects_full['year'] >= 2016) & (transects_full['year'] <= 2025)]

# Appliquer la colorisation
if 'Evolution_LRR [m/an]' in df_period.columns:
    df_period['couleur'] = df_period.apply(
        lambda row: couleur_evolution(row['Evolution_LRR [m/an]'], row['LRR']), axis=1
    )
else:
    df_period['couleur'] = 'gray'

# --- 3. Créer la carte avec les transects colorés ---
fig, ax = plt.subplots(1, 1, figsize=(15, 12))

# Afficher la baseline en arrière-plan
if not baseline_gdf.empty:
    baseline_gdf.plot(ax=ax, color='black', linewidth=1, alpha=0.7, label='Baseline')

# Afficher le trait de côte fusionné
if not fusion_trait_gdf.empty:
    fusion_trait_gdf.plot(ax=ax, color='darkblue', linewidth=2, alpha=0.8, label='Trait de côte')

# Afficher les transects colorés selon l'évolution
df_period.plot(ax=ax, color=df_period['couleur'], linewidth=2, alpha=0.8)

# --- 4. Ajouter un fond de carte (optionnel) ---
try:
    # Convertir en Web Mercator pour contextily
    df_web_mercator = df_period.to_crs(epsg=2975)
    baseline_web_mercator = baseline_gdf.to_crs(epsg=2975)
    fusion_web_mercator = fusion_trait_gdf.to_crs(epsg=3857)

    # Redessiner les couches en Web Mercator
    fig, ax = plt.subplots(1, 1, figsize=(15, 12))

    if not baseline_web_mercator.empty:
        baseline_web_mercator.plot(ax=ax, color='black', linewidth=1, alpha=0.7, label='Baseline')

    if not fusion_web_mercator.empty:
        fusion_web_mercator.plot(ax=ax, color='darkblue', linewidth=2, alpha=0.8, label='Trait de côte')

    df_web_mercator.plot(ax=ax, color=df_period['couleur'], linewidth=2, alpha=0.8)

    # Ajouter le fond de carte satellite
    ctx.add_basemap(ax, crs=df_period.crs, source=ctx.providers.Esri.WorldImagery, alpha=0.7)
except:
    print("Impossible d'ajouter le fond de carte. Continuez sans fond.")

# --- 5. Personnaliser la carte ---
ax.set_title('Évolution du trait de côte par transect (2016-2025)', fontsize=16, fontweight='bold')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

# --- 6. Créer une légende personnalisée ---
legend_elements = [
    Patch(facecolor='#8B0000', label='Érosion forte (< -1 m/an)'),
    Patch(facecolor='#FF0000', label='Érosion modérée (-1 à -0.017 m/an)'),
    Patch(facecolor='#FF6B6B', label='Érosion faible (-0.017 à 0 m/an)'),
    Patch(facecolor='#90EE90', label='Accrétion faible (0 à 0.0006 m/an)'),
    Patch(facecolor='#32CD32', label='Accrétion modérée (0.0006 à 0.0007 m/an)'),
    Patch(facecolor='#006400', label='Accrétion forte (> 0.0007 m/an)'),
    Patch(facecolor='gray', label='Données manquantes')
]

ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))

# --- 7. Ajuster l'affichage ---
plt.tight_layout()

# --- 8. Sauvegarder la couche avec les attributs de couleur ---
output_shapefile = os.path.join(output_dir, "transects_evolution_colores.shp")

df_period.to_file(output_shapefile)

print(f"Couche sauvegardée : {output_shapefile}")

current_step += 1

print()
print()

show_progress(current_step, total_steps, "Calcul statistique annuelles ......")

# Charger la couche shapefile avec les attributs

try:
    gdf = gpd.read_file(output_shapefile)
    logging.info(f"Shapefile chargé ({gdf.shape[0]} lignes)")
except Exception as e:
    logging.error(f"Échec du chargement : {str(e)}")
    sys.exit(1)

# Vérifier la présence des colonnes nécessaires
colonnes_requises = {'BEACH_CODE', 'year', 'LRR', 'WLR'}
if not colonnes_requises.issubset(gdf.columns):
    missing = colonnes_requises - set(gdf.columns)
    logging.error(f"Colonnes manquantes dans le shapefile : {missing}")
    sys.exit(1)

# Liste des BEACH_CODE à analyser
beach_codes_baie_saint_paul = [8253, 8254, 8255, 8256, 8257]
beach_codes_saint_benoit = [8274, 8275, 8276, 8277]
beach_codes = beach_codes_baie_saint_paul + beach_codes_saint_benoit

# Filtrer les données pour ces plages et années 2015-2025
df = gdf[
    (gdf['BEACH_CODE'].isin(beach_codes)) &
    (gdf['year'] >= 2016) &
    (gdf['year'] <= 2025)
]

output_file = os.path.join(output_dir, "statistiques_annuelles.txt")

calcul_et_afficher_statistiques_annuelles(df, output_file)

print(f"Statistique sauvegardé : {output_shapefile}")

current_step += 1

print()
print()

show_progress(current_step, total_steps, "Calcul statistique moyenne ......")

output_file = "resultats_statistiques_moyennes.txt"
output_path = os.path.join(output_dir, output_file)

with open(output_path, "w") as f:
    calcul_moyennes_par_groupe(df, beach_codes_baie_saint_paul, file=f)
    calcul_moyennes_par_groupe(df, beach_codes_saint_benoit, file=f)

print(f"Statistique sauvegardé : {output_shapefile}")

current_step += 1

print()
print()

logging.info("Traitement terminé ..............")
