import fnmatch
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



def recursiveFileSearch(rootdir=None, pattern='*'):
    '''
    Recherche récursive de fichiers selon un motif.

    Arguments:
    - rootdir (str): chemin du dossier racine où commencer la recherche (par défaut dossier du script)
    - pattern (str): motif des fichiers à rechercher

    Retourne:
    - liste des chemins absolus des fichiers trouvés
    '''
    if rootdir is None:
        rootdir = os.path.dirname(os.path.abspath(__file__))

    matches = []
    for root, dirnames, filenames in os.walk(rootdir):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(str(pathlib.Path(os.path.join(root, filename))))
    return matches

def createBasicFolderStructure(fs=None, base_path=''):
    '''
    Crée la structure de dossiers à partir d'un dictionnaire.

    Arguments:
    - fs (dict): structure des dossiers à créer
    - base_path (str): chemin racine où créer la structure
    '''
    if fs is None:
        fs = {
            "input_data": {
                "sds": {},
                "bundary_reunion": {}
            },
            "aux_data": {},
            "output_data": {}
        }
    for folder, subfolder in fs.items():
        folder_path = os.path.join(base_path, folder)
        os.makedirs(folder_path, exist_ok=True)
        if subfolder:
            createBasicFolderStructure(subfolder, folder_path)

def findIntoFolderStructure(fs=None, base_path='', folder_name=''):
    '''
    Trouve le chemin d'un dossier spécifique dans la structure.

    Arguments:
    - fs (dict): structure des dossiers
    - base_path (str): chemin racine
    - folder_name (str): nom du dossier à trouver

    Retourne:
    - chemin du dossier trouvé ou None
    '''
    if fs is None:
        fs = {
            "input_data": {
                "sds": {},
                "bundary_reunion": {}
            },
            "aux_data": {},
            "output_data": {}
        }
    for folder, subfolder in fs.items():
        folder_path = os.path.join(base_path, folder)
        if folder == folder_name:
            return folder_path
        if subfolder:
            result = findIntoFolderStructure(subfolder, folder_path, folder_name)
            if result:
                return result
    return None

def show_progress(current_step, total_steps, message):
    percent = int(100 * current_step / total_steps)
    now = datetime.now().strftime("%d/%m/%Y || %H:%M:%S")
    print(f"{now} || [{percent:3d}%] {message}")

def extract_date_from_filename(filename):
    # ------------------ Cherche un motif 8 chiffres suivis de 'T' puis 6 chiffres (heure) par exemple: SMIL2A_20221205T065605_ ------------------
    match = re.search(r'(\d{8})T\d{6}', filename)
    if match:
        return match.group(1)  # ------------------ Extrait uniquement la date YYYYMMDD ------------------
    else:
        return pd.NA  # ------------------ Valeur manquante si pas trouvé ------------------


# Fonction de classification

def classer_groupe(beach_code):
    if beach_code in plages_sable_blanc_sans_recif:
        return "Plage sable blanc sans récif"  # Plage sable blanc sans récif
    elif beach_code in plages_sable_blanc_avec_recif:
        return "Plage sable blanc avec récif"  # Plage sable blanc avec récif
    elif beach_code in plages_sable_noir_sans_recif:
        return "Plage sable noir sans récif"  # Plage sable noir sans récif
    elif beach_code in plages_sable_noir_avec_recif:
        return "Plage sable noir avec récif"  # Plage sable noir avec récif
    elif beach_code in plages_mixtes_sans_recif:
        return "Plage mixte sans récif"  # Plage mixte sans récif
    elif beach_code in plages_galets:
        return "Plage à galets"  # Plage à galets
    elif beach_code in embouchures:
        return "Embouchure"  # Embouchure
    else:
        return "Autres / non classé"  # Autres / non classé

# ------------------ Fonction pour extraire une contour de la couche polygone ------------------

def extract_lines(geometry):
    """Récupère tous les LineString d'une géométrie (même imbriqués)."""
    if geometry.is_empty:
        return []
    elif isinstance(geometry, LineString):
        return [geometry]
    elif isinstance(geometry, MultiLineString):
        return list(geometry.geoms)
    elif isinstance(geometry, GeometryCollection):
        lines = []
        for geom in geometry.geoms:
            lines.extend(extract_lines(geom))
        return lines
    else:
        return []


def extract_exterior_from_lines(path_lines, output_path):
    gdf = gpd.read_file(path_lines)
    lines = [geom for geom in gdf.geometry if not geom.is_empty]

    # ------------------ Polygoniser les lignes pour créer des polygones fermés ------------------

    polygons = list(polygonize(lines))
    if not polygons:
        print("Aucun polygone créé, vérifie tes lignes.")
        return

    # ------------------ Fusionner tous les polygones en un seul (peut être MultiPolygon) ------------------
    union_poly = unary_union(polygons)

    # ------------------ Extraire les contours extérieurs ------------------
    exteriors = []
    if isinstance(union_poly, Polygon):
        exteriors.append(LineString(union_poly.exterior.coords))
    elif isinstance(union_poly, MultiPolygon):
        for poly in union_poly.geoms:
            exteriors.append(LineString(poly.exterior.coords))
    else:
        print(f"Type de géométrie inattendu : {type(union_poly)}")
        return

    # ------------------ Créer GeoDataFrame et sauvegarder ------------------
    gdf_exterior = gpd.GeoDataFrame(geometry=exteriors, crs=gdf.crs)
    gdf_exterior.to_file(output_path)
    print(f"Bordure externe sauvegardée dans : {output_path}")

def flatten_geometries(geoms):
    flat = []
    for geom in geoms:
        if geom.is_empty:
            continue
        if isinstance(geom, LineString):
            flat.append(geom)
        elif isinstance(geom, (MultiLineString, GeometryCollection)):
            flat.extend(flatten_geometries(geom.geoms))
    return flat

def azimut(transect):
    # Si MultiLineString, prendre la première ligne
    try:
        if isinstance(transect, MultiLineString):
            transect = transect.geoms[0]  # première géométrie simple

        # Maintenant transect est un LineString
        x1, y1 = transect.coords[0]
        x2, y2 = transect.coords[-1]

        angle = np.degrees(np.arctan2((x2 - x1), (y2 - y1)))
        return angle % 360
    except Exception:
        return np.nan  # ou une valeur par défaut

def extract_open_line_from_closed(
    path_closed_lines: str,
    path_manual_lines: str,
    output_path: str,
    buffer_distance: float = 10.0
):
    # ------------------ Charger les couches ------------------

    closed_lines = gpd.read_file(path_closed_lines)
    manual_lines = gpd.read_file(path_manual_lines)

    # ------------------ Harmoniser les CRS ------------------

    if closed_lines.crs != manual_lines.crs:
        manual_lines = manual_lines.to_crs(closed_lines.crs)

    # ------------------ Créer un buffer autour de la ligne manuelle ------------------

    manual_buffer = manual_lines.buffer(buffer_distance)

    # ------------------ Fusionner les buffers en un seul polygone ------------------

    buffer_union = manual_buffer.union_all()

    # ------------------ Découper la couche fermée avec le buffer ------------------

    clipped = closed_lines.clip(buffer_union)

    # ------------------ Aplatir les géométries multiparties ------------------

    flat_lines = flatten_geometries(clipped.geometry)

    # ------------------ Fusionner les lignes en une seule ligne ouverte ------------------

    merged = linemerge(flat_lines)

    # ------------------ Créer GeoDataFrame avec la ligne fusionnée ------------------

    gdf_out = gpd.GeoDataFrame(geometry=[merged], crs=closed_lines.crs)

    # ------------------ Champs attributaire de la couche baseline automatiques ------------------ Réf: https://pubs.usgs.gov/of/2018/1179/ofr20181179.pdf

    if 'OBJECTED' not in gdf_out.columns:
                    gdf_out['OBJECTED'] = range(1, len(gdf_out) + 1)

    #gdf_out['OBJECTED'] = 0  # ou 'OBJECTID' selon ton besoin
    gdf_out['SHAPE'] = gdf_out.geometry.geom_type  # ou 'Shape' si tu préfères

    # ------------------ ID : [long integer] default 1 ------------------

    gdf_out['ID'] = 1

    # ------------------ Offshore : [short integer] default 0 ------------------

    gdf_out['OFFshore'] = 0 # transects sont tous côtiers, mets 0. Sinon, indique 1

    # ------------------ CastDir : valeur par défaut (exemple 0) ------------------

    gdf_out['CastDir'] = gdf_out['geometry'].apply(azimut).astype(int)

    # ------------------ shape_length : [short integer] ------------------

    gdf_out['shape_length'] = gdf_out.geometry.length

    # ------------------ Enregistrer la couche avec les nouveaux attributs ------------------

    gdf_out.to_file(output_path)

    print(f"Ligne découpée sauvegardée avec attributs dans : {output_path}")

def unit_vector(pt1, pt2):
    dx = pt2.x - pt1.x
    dy = pt2.y - pt1.y
    norm = np.sqrt(dx*dx + dy*dy)
    if norm == 0:
        return (0, 0)  # ------------------ ou un vecteur par défaut ------------------
    return (dx / norm, dy / norm)


def create_transects(baseline, n=N_TRANSECTS, length=TRANSECT_LENGTH):
    transects = []
    total_length = baseline.length
    distances = np.linspace(0, total_length, n)
    for d in distances:
        point_on_baseline = baseline.interpolate(d)
        delta = 1
        next_point = baseline.interpolate(min(d + delta, total_length))
        ux, uy = unit_vector(point_on_baseline, next_point)
        if ux == 0 and uy == 0:
            # Ignorer ce transect car direction indéfinie
            continue
        px, py = -uy, ux
        start = Point(point_on_baseline.x + px * length / 2, point_on_baseline.y + py * length / 2)
        end = Point(point_on_baseline.x - px * length / 2, point_on_baseline.y - py * length / 2)
        transects.append(LineString([start, end]))
    return transects

# ------------------ SCE (Shoreline Change Enveloppe): distance measurement ------------------

def calc_SCE(distances):
    if distances.isnull().all():
        return np.nan
    return distances.max() - distances.min()

# ------------------ NSM (Net Shoreline Movement) : distance measurement ------------------

def calc_NSM(distances):
    if distances.isnull().all():
        return np.nan
    return distances.iloc[-1] - distances.iloc[0]

# ------------------ EPR (End Oiubt Rate): Point change ------------------

def calc_EPR(nsm, years):
    if years == 0 or pd.isna(nsm):
        return np.nan
    return nsm / years

# LRR (Linear Regression Rate) : Regression statistics ------------------

def calc_LRR(dates, distances):
    if len(dates) < 2:
        return np.nan
    x = (dates - dates.min()).dt.days
    y = distances
    if y.isnull().all():
        return np.nan
    slope, intercept = np.polyfit(x, y, 1)
    return slope

# ------------------ WLR (Weighted Linear Regression) : Regression statistics ------------------

def calc_WLR(dates, distances, weights=None):
    if len(dates) < 2:
        return np.nan
    x = (dates - dates.min()).dt.days.values.reshape(-1, 1)
    y = distances.values
    w = weights.values if weights is not None else np.ones_like(y)
    x = sm.add_constant(x)
    model = sm.WLS(y, x, weights=w)
    results = model.fit()
    return results.params[1]  # pente

# ------------------ LMS (Least Median of Squares) : advanced statics ------------------

def calc_LMS(dates, distances):
    if len(dates) < 2:
        return np.nan
    x = (dates - dates.min()).dt.days.values.reshape(-1, 1)
    y = distances.values
    model = RANSACRegressor(base_estimator=LinearRegression(), min_samples=2)
    model.fit(x, y)
    slope = model.estimator_.coef_[0]
    return slope

def classify_evolution(rate, seuil=1e-4):
    if pd.isnull(rate):
        return 'NA'
    elif lrr < 0:
        return 'érosion'
    elif lrr > 0:
        return 'accrétion'
    else:
        return 'NA'

# --- 2. Définir la palette de couleurs pour la classification ---
def couleur_evolution(evol, lrr_value):
    if pd.isnull(evol) or pd.isnull(lrr_value):
        return 'gray'
    elif evol.lower() == 'érosion':
        # Gradation de rouge selon l'intensité d'érosion
        if lrr_value < -1:
            return '#8B0000'  # Rouge foncé
        elif lrr_value < -0.017:
            return '#FF0000'  # Rouge
        else:
            return '#FF6B6B'  # Rouge clair
    elif evol.lower() == 'accrétion':
        # Gradation de bleu/vert selon l'intensité d'accrétion
        if lrr_value > 0.0006:
            return '#006400'  # Vert foncé
        elif lrr_value > 0.0007:
            return '#32CD32'  # Vert
        else:
            return '#90EE90'  # Vert clair
    else:
        return 'gray'

def calcul_et_afficher_statistiques_annuelles(df, output_path):
    # Définition des paramètres
    largeur = 50  # Largeur constante en mètres
    annees = range(2016, 2026)

    # Définition des groupes de plages avec leurs longueurs
    groupes = {
        'baie_saint_paul': {
            'codes': [8253, 8254, 8255, 8256, 8257],
            'longueur': 16520  # Longueur totale pour le groupe
        },
        'saint_benoit': {
            'codes': [8274, 8275, 8276, 8277],
            'longueur': 7480  # Longueur totale pour le groupe
        }
    }

    with open(output_path, 'w') as f, redirect_stdout(f):
        print(f"{'Année':<6} {'Groupe':<15} {'Taux LRR Érosion (m/an)':>25} {'Taux WLR Érosion (m/an)':>25} "
              f"{'Vol. Érosion LRR (m³/an)':>25} {'Vol. Érosion WLR (m³/an)':>25} "
              f"{'Vol. Accrétion LRR (m³/an)':>27} {'Vol. Accrétion WLR (m³/an)':>27}")
        print("-" * 160)

        # Calcul pour chaque année et chaque groupe
        for annee in annees:
            print(f"\n=== Année {annee} ===")

            for groupe_nom, config in groupes.items():
                # Filtrer les données pour l'année et le groupe
                mask = (df['BEACH_CODE'].isin(config['codes'])) & (df['year'] == annee)
                df_annee = df[mask]

                # Calcul des taux d'érosion moyens
                lrr_erosion = df_annee[df_annee['LRR'] < 0]['LRR'].mean()
                wlr_erosion = df_annee[df_annee['WLR'] < 0]['WLR'].mean()

                lrr_accretion = df_annee[df_annee['LRR'] > 0]['LRR'].mean()
                wlr_accretion = df_annee[df_annee['WLR'] > 0]['WLR'].mean()

                # Calcul des volumes de sédiments
                volume_erosion_lrr = (-lrr_erosion * largeur * config['longueur']) if not np.isnan(lrr_erosion) else 0
                volume_erosion_wlr = (-wlr_erosion * largeur * config['longueur']) if not np.isnan(wlr_erosion) else 0

                volume_accretion_lrr = (lrr_accretion * largeur * config['longueur']) if not np.isnan(lrr_erosion) else 0
                volume_accretion_wlr = (wlr_accretion * largeur * config['longueur']) if not np.isnan(wlr_erosion) else 0

                print(f"{annee:<6} {groupe_nom:<15} "
                      f"{lrr_erosion:25.6f} {wlr_erosion:25.6f} "
                      f"{volume_erosion_lrr:25.2f} {volume_erosion_wlr:25.2f} "
                      f"{volume_accretion_lrr:27.2f} {volume_accretion_wlr:27.2f}")

    print(f"Résultats enregistrés dans : {output_path}")

# Fonction show_progress vide (pas d'affichage)
def show_progress(current_step, total_steps, message):
    pass

def calcul_moyennes_par_groupe(df, codes_plage, file=None, largeur=50):
    # Filtrer les données pour le groupe de plages
    df_groupe = df[df['BEACH_CODE'].isin(codes_plage)]

    # Moyennes des taux LRR et WLR
    mean_LRR = df_groupe['LRR'].mean()
    mean_WLR = df_groupe['WLR'].mean()

    # Taux d’érosion moyen (valeurs négatives)
    erosion_LRR = df_groupe.loc[df_groupe['LRR'] < 0, 'LRR']
    erosion_WLR = df_groupe.loc[df_groupe['WLR'] < 0, 'WLR']
    mean_erosion_LRR = erosion_LRR.mean() if not erosion_LRR.empty else np.nan
    mean_erosion_WLR = erosion_WLR.mean() if not erosion_WLR.empty else np.nan

    # Taux d’accrétion moyen (valeurs positives)
    accretion_LRR = df_groupe.loc[df_groupe['LRR'] > 0, 'LRR']
    accretion_WLR = df_groupe.loc[df_groupe['WLR'] > 0, 'WLR']
    mean_accretion_LRR = accretion_LRR.mean() if not accretion_LRR.empty else np.nan
    mean_accretion_WLR = accretion_WLR.mean() if not accretion_WLR.empty else np.nan

    # Estimer la longueur totale des plages du groupe (somme des longueurs individuelles)
    # Ici on utilise la même longueur par plage selon les codes, à adapter si tu as les longueurs exactes
    longueur_totale = 0
    for code in codes_plage:
        if code in beach_codes_baie_saint_paul:
            longueur_totale += 16520
        elif code in beach_codes_saint_benoit:
            longueur_totale += 7480
        else:
            longueur_totale += 0  # ou valeur par défaut

    # Quantité sédimentaire érodée et accrétionnée (volume en m3/an)
    volume_erosion = (-erosion_LRR * largeur * longueur_totale).sum() if not erosion_LRR.empty else 0
    volume_accretion = (accretion_LRR * largeur * longueur_totale).sum() if not accretion_LRR.empty else 0

    # Écart-type et Kappa moyen pour LRR et WLR
    std_LRR = df_groupe['LRR'].std()
    std_WLR = df_groupe['WLR'].std()

    # Pour Kappa, si tu as les valeurs par plage, calcule la moyenne ici
    # Sinon, calcule globalement sur le groupe (exemple simplifié)
    # Ici on calcule Kappa entre LRR et WLR binarisés par moyenne globale
    global_mean_LRR = mean_LRR
    global_mean_WLR = mean_WLR
    y_pred_LRR = (df_groupe['LRR'] >= global_mean_LRR).astype(int)
    y_pred_WLR = (df_groupe['WLR'] >= global_mean_WLR).astype(int)
    kappa = cohen_kappa_score(y_pred_LRR, y_pred_WLR)

    # Affichage des résultats
    print(f"--- Statistiques pour le groupe de plages {codes_plage} ---", file=file)
    print(f"Taux moyen LRR : {mean_LRR:.6f} m/an", file=file)
    print(f"Taux moyen WLR : {mean_WLR:.6f} m/an", file=file)
    print(f"Taux d'érosion moyen LRR : {mean_erosion_LRR:.6f} m/an", file=file)
    print(f"Taux d'accrétion moyen LRR : {mean_accretion_LRR:.6f} m/an", file=file)
    print(f"Taux d'érosion moyen WLR : {mean_erosion_WLR:.6f} m/an", file=file)
    print(f"Taux d'accrétion moyen WLR : {mean_accretion_WLR:.6f} m/an", file=file)
    print(f"Quantité sédiment érodée (LRR) : {volume_erosion:.2f} m3/an", file=file)
    print(f"Quantité sédiment accrétionnée (LRR) : {volume_accretion:.2f} m3/an", file=file)
    print(f"Écart-type LRR : {std_LRR:.6f}", file=file)
    print(f"Écart-type WLR : {std_WLR:.6f}", file=file)
    print(f"Kappa moyen entre LRR et WLR (binarisés) : {kappa:.4f}", file=file)
    print("-------------------------------------------------------------\n", file=file)
