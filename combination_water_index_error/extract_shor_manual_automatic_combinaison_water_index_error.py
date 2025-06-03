import os
import glob
import numpy as np
import pathlib
import logging
import csv
import json
import shapefile
import fnmatch
import networkx as nx
import gc
from osgeo import gdal, ogr, osr
from skimage.filters import threshold_otsu, threshold_multiotsu
from skimage import morphology
from numpy import sqrt
from datetime import datetime
from math import sqrt, exp
from tqdm import tqdm
from skimage.transform import rescale
from matplotlib.pyplot import clf, contour
from scipy.spatial import Delaunay


# _____________________ PATH _____________________

output_dir = "/home/jonathan/SAET/SAET_installation/"

# _____________________ Output de la série temporelle S2 _____________________

root_path = "/home/jonathan/SAET/SAET_installation/Test/"


def createFolderCheck(folder_path):
    '''
    Description:
    ------------
    Creates a new folder only in case this folder does not exist

    Arguments:
    ------------
    - folder_path (string): folder to be created

    Returns:
    ------------
    None

    '''
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def getRasterEPSG(raster_path):
    ds = gdal.Open(raster_path)
    srs = osr.SpatialReference(wkt=ds.GetProjection())
    try:
        return srs.GetAttrValue("AUTHORITY", 1)
    except:
        return None

def polyfit2d(x, y, z, order=2):
    """
    Ajuste un polynôme 2D d'ordre donné aux points (x, y, z).
    Retourne les coefficients.
    """
    import numpy as np
    ncols = (order + 1) ** 2
    G = np.zeros((x.size, ncols))
    ij = [(i, j) for i in range(order + 1) for j in range(order + 1)]
    for index, (i, j) in enumerate(ij):
        G[:, index] = x**i * y**j
    m, _, _, _ = np.linalg.lstsq(G, z, rcond=None)
    return m

def polyval2d(x, y, m, order=2):
    """
    Evalue le polynôme 2D d'ordre donné en (x, y) avec les coefficients m.
    """
    import numpy as np
    ij = [(i, j) for i in range(order + 1) for j in range(order + 1)]
    z = np.zeros_like(x, dtype=float)
    for a, (i, j) in zip(m, ij):
        z += a * x**i * y**j
    return z

def saveIndex(in_array, out, template_path, dType=gdal.GDT_Float32):
    '''
    Description:
    ------------
    Saves water index to tiff image

    Arguments:
    ------------
    - in_array (numpy matrix): water index data
    - out (string): output path to the tiff image
    - template_path (string): template image to copy resolution, bounding box
    and coordinate reference system.
    - dType: data type format (default: float 32 bits)

    Returns:
    ------------
    None

    '''

    if os.path.exists(out):
        os.remove(out)

    template = gdal.Open(template_path)
    driver = gdal.GetDriverByName('GTiff')
    shape = in_array.shape
    dst_ds = driver.Create(
        out, xsize=shape[1], ysize=shape[0], bands=1, eType=dType)
    proj = template.GetProjection()
    geo = template.GetGeoTransform()
    dst_ds.SetGeoTransform(geo)
    dst_ds.SetProjection(proj)
    dst_ds.GetRasterBand(1).WriteArray(in_array)
    dst_ds.FlushCache()
    dst_ds = None

from osgeo import gdal

def check_raster_alignment(raster1, raster2):
    ds1 = gdal.Open(raster1)
    ds2 = gdal.Open(raster2)
    assert ds1.RasterXSize == ds2.RasterXSize, f"Largeur différente : {ds1.RasterXSize} vs {ds2.RasterXSize}"
    assert ds1.RasterYSize == ds2.RasterYSize, f"Hauteur différente : {ds1.RasterYSize} vs {ds2.RasterYSize}"
    assert ds1.GetGeoTransform() == ds2.GetGeoTransform(), f"Géotransformée différente : {ds1.GetGeoTransform()} vs {ds2.GetGeoTransform()}"
    assert ds1.GetProjection() == ds2.GetProjection(), "Projection différente"
    print("✅ Les deux rasters sont parfaitement alignés.")


def mask_raster_with_shapefile(raster_path, shapefile_path, out_path,
                               ref_bounds, ref_xsize, ref_ysize, ref_proj):
    """
    Découpe (masque) un raster à la zone d'un shapefile.
    """
    options = gdal.WarpOptions(
        cutlineDSName=shapefile_path,
        cropToCutline=True,
        dstNodata=0,
        outputBounds=ref_bounds,
        width=ref_xsize,
        height=ref_ysize,
        dstSRS='EPSG:2975' # Reprojection au cordonnée géographique de la Réunion EPSG:2975 - RGR92 / UTM zone 40S
    )
    gdal.Warp(out_path, raster_path, options=options)
    return out_path

def getSourceEpsg(raster_or_vector_path):
    """
    Retourne le code EPSG (int) du raster ou shapefile en entrée.
    """
    # Pour un raster
    try:
        ds = gdal.Open(raster_or_vector_path)
        if ds is not None:
            srs = osr.SpatialReference(wkt=ds.GetProjection())
            code = srs.GetAttrValue('AUTHORITY', 1)
            if code is not None:
                return int(code)
    except:
        pass

    # Pour un shapefile
    try:
        ds = ogr.Open(raster_or_vector_path)
        if ds is not None:
            layer = ds.GetLayer()
            srs = layer.GetSpatialRef()
            code = srs.GetAttrValue('AUTHORITY', 1)
            if code is not None:
                return int(code)
    except:
        pass

    raise ValueError(f"Impossible de déterminer l'EPSG pour {raster_or_vector_path}")

from osgeo import gdal

def align_raster_to_template(input_raster, template_raster, output_raster):
    template_ds = gdal.Open(template_raster)
    geo = template_ds.GetGeoTransform()
    proj = template_ds.GetProjection()
    width = template_ds.RasterXSize
    height = template_ds.RasterYSize

    gdal.Warp(
        output_raster,
        input_raster,
        format='GTiff',
        outputBounds=(
            geo[0],
            geo[3] + geo[5] * height,
            geo[0] + geo[1] * width,
            geo[3]
        ),
        dstSRS=proj,
        width=width,
        height=height,
        resampleAlg='nearest'
    )


def getTargetEpsg(scene_path, band_name):
    """
    Retourne le code EPSG (int) de la projection de la bande spécifiée dans la scène Sentinel-2.
    """

    band_files = glob.glob(os.path.join(scene_path, f"*{band_name}*.jp2"))
    if not band_files:
        raise FileNotFoundError(f"Bande {band_name} non trouvée dans {scene_path}")
    ds = gdal.Open(band_files[0])
    srs = osr.SpatialReference(wkt=ds.GetProjection())
    code = srs.GetAttrValue('AUTHORITY', 1)
    if code is None:
        raise ValueError(f"Impossible de lire l'EPSG de {band_files[0]}")
    return int(code)


def resample_raster_to_match(src_path, ref_path, out_path):
    src_ds = gdal.Open(src_path)
    ref_ds = gdal.Open(ref_path)
    options = gdal.WarpOptions(
        format='GTiff',
        outputBounds=ref_ds.GetGeoTransform(),
        width=ref_ds.RasterXSize,
        height=ref_ds.RasterYSize,
        dstSRS=ref_ds.GetProjection(),
        resampleAlg='nearest'
    )
    gdal.Warp(out_path, src_ds, options=options)
    src_ds = None
    ref_ds = None

# Projection du raster en projection standard 4326

def reprojectRaster(input_raster_path, output_raster_path, target_epsg, resample_alg='bilinear', dst_nodata=0):
    '''
    Reprojette un raster vers n'importe quel EPSG.
    '''
    src_ds = gdal.Open(input_raster_path)
    if src_ds is None:
        raise FileNotFoundError(f"Fichier introuvable : {input_raster_path}")

    target_srs = osr.SpatialReference()
    target_srs.ImportFromEPSG(target_epsg)

    gdal.Warp(
        destNameOrDestDS=output_raster_path,
        srcDSOrSrcDSTab=src_ds,
        dstSRS=target_srs.ExportToWkt(),
        format='GTiff',
        resampleAlg=resample_alg,
        dstNodata=dst_nodata
    )
    src_ds = None
    return output_raster_path

def check_shapefile_exists(shp_path):
    base = os.path.splitext(shp_path)[0]
    required = [base + ext for ext in ['.shp', '.shx', '.dbf']]
    for f in required:
        if not os.path.isfile(f):
            raise FileNotFoundError(f"Fichier manquant pour le shapefile : {f}")


def removeHolesByArea(img, area):
    '''
    Removes little holes from binary images.
    '''
    img_closed = morphology.area_closing(img, area, connectivity=1)
    img_closed = morphology.area_closing(~img_closed, area, connectivity=1)
    return ~img_closed


def getIndexMask(index_path, thr_method, tol_area=300, manual_threshold=None):
    '''
    Description:
    ------------
    Computes binary mask from water index using the standar value 0 for
    segmentation

    Arguments:
    ------------
    - index_path (string): path to water index
    - thr_method (string): method to segmentation threshold computation
      {'0': standard zero, '1': otsu bimodal, '2': otsu multimodal with 3 clases}
    - tol_area (int): tolerance to remove small holes. Default: 300

    Returns:
    ------------
    - imgmask (numpy matrix): if area removing is enabled
    - index_copy (numpy matrix): if area removing is disabled

    '''

    if not isinstance(index_path, str):
        raise TypeError(f"index_path doit être une chaine, reçu: {type(index_path)}")

    index_ds = gdal.Open(index_path)
    band = index_ds.GetRasterBand(1)
    index_data = band.ReadAsArray()
    index_data[index_data == float('-inf')] = 0.
    index_data = np.nan_to_num(index_data)  # replace nan values by 0

    # tolerance for segmentation
    if thr_method == '0':
        if manual_threshold is not None:
            tol = manual_threshold
        else:
            tol = 0

    elif thr_method == '1':
        # Progression ligne à ligne pour Otsu
        n = index_data.shape[0]
        tol_list = []
        for i in range(n):
            vec_line = index_data[i, :]
            tol_line = threshold_otsu(vec_line)
            tol_list.append(tol_line)
            show_step_progress(i+1, n, "Seuillage Otsu ligne à ligne")

        tol = np.median(tol_list)

    if thr_method == '2':
        n = index_data.shape[0]
        tol_list = []
        for i in range(n):
            line = index_data[i, :]
            th_otsu_multi = threshold_multiotsu(line, 3)

            tol_line = th_otsu_multi[0] if abs(th_otsu_multi[0]) < abs(th_otsu_multi[1]) else th_otsu_multi[1]

            tol_list.append(tol_line)
            show_step_progress(i+1, n, "Seuillage Otsu multimodal ligne à ligne")

        tol = np.median(tol_list)

    # image binarization according threshold
    index_copy = index_data.copy()
    index_copy[index_data < tol] = 0.  # land
    index_copy[index_data >= tol] = 1.  # water

    if tol_area != 0:  # area removing
        img_mask = removeHolesByArea(index_copy.astype(np.byte), tol_area)
        return img_mask
    else:
        return index_copy

def raster_to_shapefile(raster_path, shp_path, epsg=2975):
    """
    Convertit un raster binaire en shapefile de polygones
    """
    src_ds = gdal.Open(raster_path)
    srcband = src_ds.GetRasterBand(1)
    drv = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(shp_path):
        drv.DeleteDataSource(shp_path)
    dst_ds = drv.CreateDataSource(shp_path)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    dst_layer = dst_ds.CreateLayer("polygons", srs=srs)
    fd = ogr.FieldDefn("DN", ogr.OFTInteger)
    dst_layer.CreateField(fd)
    gdal.Polygonize(srcband, None, dst_layer, 0, [], callback=None)
    dst_ds = None

def getBandData(band_path):
    '''
    Returns the data matrix from a band path
    '''
    band = gdal.Open(band_path)
    band_data = band.GetRasterBand(1).ReadAsArray()
    return band_data

def add_cpg_file(shp_path, encoding="UTF-8"):
    cpg_path = os.path.splitext(shp_path)[0] + ".cpg"
    with open(cpg_path, "w") as f:
        f.write(encoding)


def downScaling(band_path):
    src = gdal.Open(band_path)
    geo = src.GetGeoTransform()
    proj = src.GetProjection()
    # On ne spécifie QUE la résolution cible, pas la taille !
    options = gdal.WarpOptions(
        format='MEM',
        xRes=20.0,
        yRes=20.0,
        resampleAlg=gdal.GRA_Bilinear
    )
    ds_resampled = gdal.Warp('', src, options=options)
    if ds_resampled is None:
        raise RuntimeError(f"gdal.Warp a échoué pour {band_path}")
    band = ds_resampled.GetRasterBand(1)
    array = band.ReadAsArray().astype(np.float32)
    return array

def show_progress(current_step, total_steps, message):
    percent = int(100 * current_step / total_steps)
    now = datetime.now().strftime("%d/%m/%Y || %H:%M:%S")
    print(f"{now} || [{percent:3d}%] {message}")

def show_step_progress(step, total, label=""):
    percent = int(100 * step / total)
    now = datetime.now().strftime("%d/%m/%Y || %H:%M:%S")
    print(f"\r{now} || {label} [{percent:3d}%]", end="")
    if step == total:
        print()


def getBandData20m(band_path):
    band = gdal.Open(band_path)
    if band is None:
        raise FileNotFoundError(f"Impossible d'ouvrir le raster {band_path}")
    pix_size = band.GetGeoTransform()[1]
    if pix_size == 10.0:
        return downScaling(band_path)
    else:
        return band.GetRasterBand(1).ReadAsArray().astype(np.float32)

def recursiveFileSearch(rootdir='.', pattern='*'):
    '''
    Description:
    ------------
    search for files recursively based in pattern strings

    Arguments:
    ------------
    - rootdir (string): path to the base folder
    - pattern (string): pattern to search files

    Returns:
    ------------
    - matches (list of strings): list of absolute paths to each found file

    '''

    matches = []
    for root, dirnames, filenames in os.walk(rootdir):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(str(pathlib.Path(os.path.join(root, filename))))
    return matches

def getBandPath(scene_path, band_name):
    '''
    Description:
    ------------
    Get absolute path from a single band or file

    Arguments:
    ------------
    - scene_path (string): path to the target folder
    - band_name (string): band name to search

    Returns:
    ------------
    - band_path (string): path to the band

    '''
    file_list = recursiveFileSearch(scene_path, '*.*')
    band_path = [i for i in file_list if (
        band_name in i) and (not 'xml' in i)]
    if len(band_path) != 0:
        return str(pathlib.Path(band_path[0]))
    else:
        return None

def getBandData20mFromScene(scene_path, band_name):
    """
    Cherche le chemin de la bande puis lit la donnée à 20m (resample si besoin).
    """
    band_path = getBandPath(scene_path, band_name)
    if band_path is None:
        raise FileNotFoundError(f"Bande {band_name} non trouvée dans {scene_path}")
    return getBandData20m(band_path)


def calc_MNDWI(band_green, band_swir1):
    return (band_green - band_swir1) / (band_green + band_swir1 + 1e-6)

def process_mndwi(data_green, data_swir1, temp_dir, date_folder, template_band):
    mndwi = calc_NDWI(data_green, data_swir1)

    mndwi_tif = os.path.join(temp_dir, f"{date_folder}_MNDWI.tif")
    saveIndex(mndwi, mndwi_tif, template_band)

    mndwi_mask = getIndexMask(mndwi_tif, thr_method='0', manual_threshold=0.1)

    mndwi_mask_tif = os.path.join(temp_dir, f"{date_folder}_MNDWI_mask.tif")
    saveIndex(mndwi_mask, mndwi_mask_tif, template_band, dType=gdal.GDT_Byte)

    return mndwi, mndwi_mask, mndwi_tif, mndwi_mask_tif

def calc_AWEINSH(band_green, band_nir, band_swir1, band_swir2):
    return 4 * (band_green - band_swir1) - 0.25 * (band_nir + (2.75 *band_swir2))

'''def calc_AWEINSH(band_green, band_nir, band_swir1, band_swir2):

    band_green = np.asarray(band_green, dtype=np.float32)
    band_nir = np.asarray(band_nir, dtype=np.float32)
    band_swir1 = np.asarray(band_swir1, dtype=np.float32)
    band_swir2 = np.asarray(band_swir2, dtype=np.float32)
    return 4 * (band_green - band_swir1) - 0.25 * (band_nir + (2.75 *band_swir2))'''

def process_aweinsh(data_green, data_nir, data_swir1, data_swir2, date_folder, template_band):
    aweinsh = calc_AWEINSH(data_green, data_nir, data_swir1, data_swir2)

    aweinsh_tif = os.path.join(temp_dir, f"{date_folder}_AWEINSH.tif")
    saveIndex(aweinsh, aweinsh_tif, template_band)

    aweinsh_mask = getIndexMask(aweinsh_tif, thr_method='0')

    aweinsh_mask_tif = os.path.join(temp_dir, f"{date_folder}_AWEINSH_mask.tif")
    saveIndex(aweinsh_mask, aweinsh_mask_tif, template_band, dType=gdal.GDT_Byte)

    return aweinsh, aweinsh_mask, aweinsh_tif, aweinsh_mask_tif


def calc_AWEISH(band_blue, band_green, band_nir, band_swir1, band_swir2):
    return band_blue + (1.5 * band_green) - 1.5 * (band_nir + band_swir1) - (0.25 * band_swir2)

def process_aweish(data_blue, data_green, data_nir, data_swir1, data_swir2, date_folder, template_band):
    aweish = calc_AWEISH(data_blue, data_green, data_nir, data_swir1, data_swir2)

    aweish_tif = os.path.join(temp_dir, f"{date_folder}_AWEISH.tif")
    saveIndex(aweish, aweish_tif, template_band)

    aweish_mask = getIndexMask(aweish_tif, thr_method='1')

    aweish_mask_tif = os.path.join(temp_dir, f"{date_folder}_AWEISH_mask.tif")
    saveIndex(aweish_mask, aweish_mask_tif, template_band, dType=gdal.GDT_Byte)

    return aweish, aweish_mask, aweish_tif, aweish_mask_tif


def calc_NDWI(band_green, band_nir):
    return (band_green - band_nir) / (band_green + band_nir + 1e-6)

def process_ndwi(data_green, data_nir, temp_dir, date_folder, template_band):
    ndwi = calc_NDWI(data_green, data_nir)

    ndwi_tif = os.path.join(temp_dir, f"{date_folder}_NDWI.tif")
    saveIndex(ndwi, ndwi_tif, template_band)

    ndwi_mask = getIndexMask(ndwi_tif, thr_method='0', manual_threshold=0.20)

    ndwi_mask_tif = os.path.join(temp_dir, f"{date_folder}_NDWI_mask.tif")
    saveIndex(ndwi_mask, ndwi_mask_tif, template_band, dType=gdal.GDT_Byte)

    return ndwi, ndwi_mask, ndwi_tif, ndwi_mask_tif


def calc_NDVI(band_nir, band_red):
    return (band_nir - band_red) / (band_nir + band_red + 1e-6) # + 1e-6: pour éviter la division par 0

def process_ndvi(data_red, data_nir, temp_dir, date_folder, template_band):
    ndvi = calc_NDVI(data_nir, data_red)

    ndvi_tif = os.path.join(temp_dir, f"{date_folder}_NDVI.tif")
    saveIndex(ndvi, ndvi_tif, template_band)

    ndvi_mask = getIndexMask(ndvi_tif, thr_method='0')

    ndvi_mask_tif = os.path.join(temp_dir, f"{date_folder}_NDVI_mask.tif")
    saveIndex(ndvi_mask, ndvi_mask_tif, template_band, dType=gdal.GDT_Byte)

    return ndvi, ndvi_mask, ndvi_tif, ndvi_mask_tif


def calc_IB(band_red, band_nir):
    return np.sqrt((band_red)**2 + (band_nir)**2)

def process_ib(data_red, data_nir, temp_dir, date_folder, template_band):
    ib = calc_IB(data_red, data_nir)

    ib_tif = os.path.join(temp_dir, f"{date_folder}_IB.tif")
    saveIndex(ib, ib_tif, template_band)

    ib_mask = getIndexMask(ib_tif, thr_method='2')

    ib_mask_tif = os.path.join(temp_dir, f"{date_folder}_IB_mask.tif")
    saveIndex(ib_mask, ib_mask_tif, template_band, dType=gdal.GDT_Byte)

    return ib, ib_mask, ib_tif, ib_mask_tif

def process_combination_mask(
    mndwi_mask, aweish_mask, aweinsh_mask, ndwi_mask, ndvi_mask, IB_mask,
    temp_dir, date_folder, template_band,
    weights=(7, 6, 5, 4, 2, 1),
    threshold=12):
    """
    Calcule la combinaison pondérée des masques, sauvegarde le raster de score et le raster binaire final.
    Retourne les chemins des fichiers créés.
    """
    # Calcul du score pondéré
    combination_index = (
        weights[0] * mndwi_mask +
        weights[1] * aweish_mask +
        weights[2] * aweinsh_mask +
        weights[3] * ndwi_mask +
        weights[4] * ndvi_mask +
        weights[5] * IB_mask
    )

    # Sauvegarde du raster de score
    comb_tif = os.path.join(temp_dir, f"{date_folder}_COMBI.tif")
    saveIndex(combination_index, comb_tif, template_band, dType=gdal.GDT_Byte)

    # Binarisation selon le seuil
    combination_mask = (combination_index >= threshold).astype(np.uint8)

    # Sauvegarde du raster binaire
    comb_tif_mask = os.path.join(temp_dir, f"{date_folder}_COMBI_mask.tif")
    saveIndex(combination_mask, comb_tif_mask, template_band, dType=gdal.GDT_Byte)

    return combination_index, combination_mask, comb_tif, comb_tif_mask


def createPixelLine(method, mask_path, cloud_mask_path, cloud_buffer=9):
    """
    Calcule le contour pixelisé (trait de côte brut) à partir d'un masque binaire (ex : combinaison pondérée)
    en retirant les zones nuageuses.

    Arguments :
    - method (str) : 'erosion' ou 'dilation' (méthode de contour)
    - mask_path (str) : chemin du raster binaire eau/terre (ex : COMBI_mask.tif)
    - cloud_mask_path (str) : chemin du raster binaire nuage (ex : *_cloudmask.tif)

    Retour :
    - pixel_line (np.ndarray) : matrice binaire du trait de côte, sans nuages
    """

    # Lecture du masque eau/terre
    mask_ds = gdal.Open(mask_path)
    mask = mask_ds.GetRasterBand(1).ReadAsArray().astype(np.uint8)

    # Calcul du contour pixelisé
    if method == 'erosion':
        erosion = morphology.binary_erosion(mask)
        pixel_line = mask - erosion
    elif method == 'dilation':
        dilation = morphology.binary_dilation(mask)
        pixel_line = dilation - mask
    else:
        raise ValueError("method doit être 'erosion' ou 'dilation'")


    # Suppression des pixels de contour dans les zones nuageuses (optionnel)
    if cloud_mask_path is not None:
        cmask_ds = gdal.Open(cloud_mask_path)
        cmask = cmask_ds.GetRasterBand(1).ReadAsArray().astype(np.uint8)
        # Vérification de la taille
        if cmask.shape != pixel_line.shape:
            raise ValueError(
                f"Taille du masque nuage {cmask.shape} différente du contour {pixel_line.shape}. "
                "Resample le masque nuage sur la grille du masque eau/terre !"
            )

        # Buffer de nuages (élargit la zone de masque nuage pour plus de sécurité)
        kernel = np.ones((cloud_buffer, cloud_buffer), np.uint8)
        cmask_dilated = morphology.binary_dilation(cmask, kernel)

        # Suppression des pixels de contour dans les zones nuageuses
        pixel_line[cmask_dilated == 1] = 0

    # Optionnel : mettre à 255 pour visualisation (sinon, laisse à 1)
    pixel_line[pixel_line == 1] = 255

    return pixel_line


def createCloudMaskS2(scene_path, output_dir, cloud_mask_level='2'):
    """
    Crée un masque binaire de nuages à partir de la bande SCL pour Sentinel-2 L2A.
    Enregistre le masque dans output_dir.
    cloud_mask_level:
        '0' : pas de masque (tout à 0)
        '1' : uniquement nuages denses (SCL=9)
        '2' : tout type de nuage (SCL=3,8,9,10)
    """
    # Cherche la bande SCL
    scl_band_path = None
    for f in os.listdir(scene_path):
        if 'SCL' in f and f.endswith('.jp2'):
            scl_band_path = os.path.join(scene_path, f)
            break
    if scl_band_path is None:
        print(f"SCL band not found in {scene_path}")
        return None

    # Valeurs SCL pour les nuages
    if cloud_mask_level == '0':
        mask_values = []
    elif cloud_mask_level == '1':
        mask_values = [9]
    elif cloud_mask_level == '2':
        mask_values = [3, 8, 9, 10]
    else:
        raise ValueError("cloud_mask_level doit être '0', '1' ou '2'")

    # Lecture de la bande SCL
    scl_ds = gdal.Open(scl_band_path)
    scl_data = scl_ds.GetRasterBand(1).ReadAsArray()

    # Création du masque
    cloud_mask = np.isin(scl_data, mask_values).astype(np.uint8)
    # Nettoyage (enlève petits objets)
    cloud_mask = morphology.remove_small_objects(cloud_mask.astype(bool), min_size=10, connectivity=1)
    cloud_mask = cloud_mask.astype(np.uint8)

    # Enregistrement
    scene_name = os.path.basename(scene_path)
    out_mask_path = os.path.join(temp_dir, f"{scene_name}_cloudmask.tif")
    template_band = scl_band_path  # même géométrie

    # Utilise ta fonction saveIndex du paste.txt
    saveIndex(cloud_mask, out_mask_path, template_band, dType=gdal.GDT_Byte)
    print(f"Masque de nuages sauvegardé : {out_mask_path}")
    return out_mask_path


def reprojectShp(input_shp, output_shp, inSpatialRef, outSpatialRef): #reprojectShp(input_shp, output_shp, in_epsg, out_epsg)
    """
    Reprojette un shapefile d'un EPSG source vers un EPSG cible.
    """
    # Définir les systèmes de coordonnées source et cible
    #inSpatialRef = osr.SpatialReference()
    #inSpatialRef.ImportFromEPSG(in_epsg)
    #outSpatialRef = osr.SpatialReference()
    #outSpatialRef.ImportFromEPSG(out_epsg)
    coordTrans = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)

    # Ouvrir le shapefile d'entrée
    #driver = ogr.GetDriverByName('ESRI Shapefile')
    #inDataSet = driver.Open(input_shp) #n’ouvre pas un shapefile en lecture, elle est destinée à la création ou à l’ouverture en écriture.

    inDataSet = ogr.Open(input_shp)
    if inDataSet is None:
        raise FileNotFoundError(f"Impossible d'ouvrir {input_shp}")

    # *** AJOUT DU SRS ***
    outLayer = outDataSet.CreateLayer(" ", geom_type=inLayer_geomtype, srs=outSpatialRef)

    # Copier la sctructure des éttributs
    inLayer = inDataSet.GetLayer()
    inLayer_geomtype = inLayer.GetGeomType()
    #inLayer.ResetReading()

    driver = ogr.GetDriverByName('ESRI Shapefile')

    # Supprimer la sortie si elle existe déjà
    if os.path.exists(output_shp):
        driver.DeleteDataSource(output_shp)

    # Vérifie que le dossier de sortie existe
    out_dir = os.path.dirname(output_shp)
    if not os.path.isdir(out_dir):
        raise FileNotFoundError(f"Le dossier de sortie n'existe pas : {out_dir}")

    print(f"Création du shapefile de sortie : {output_shp}")

    outDataSet = driver.CreateDataSource(output_shp)
    outLayer = outDataSet.CreateLayer(" ", geom_type=inLayer_geomtype)

    # Copier la structure des attributs
    inLayerDefn = inLayer.GetLayerDefn()
    for i in range(0, inLayerDefn.GetFieldCount()):
        fieldDefn = inLayerDefn.GetFieldDefn(i)
        outLayer.CreateField(fieldDefn)

    outLayerDefn = outLayer.GetLayerDefn()

    # Copier les géométries et attributs
    inFeature = inLayer.GetNextFeature()
    while inFeature:
        geom = inFeature.GetGeometryRef()
        if geom is not None:
            geom.Transform(coordTrans)
            outFeature = ogr.Feature(outLayerDefn)
            outFeature.SetGeometry(geom)
            for i in range(0, outLayerDefn.GetFieldCount()):
                outFeature.SetField(outLayerDefn.GetFieldDefn(i).GetNameRef(), inFeature.GetField(i))
            outLayer.CreateFeature(outFeature)
            outFeature = None
        inFeature = inLayer.GetNextFeature()

    inLayer = None
    inDataSet = None
    outLayer = None
    outDataSet = None

    # Créer le fichier .prj
    outSpatialRef.MorphToESRI()
    file_name = os.path.basename(output_shp)
    dir_name = os.path.dirname(output_shp)
    prj_name = str(pathlib.Path(os.path.join(dir_name, file_name.split('.')[0] + '.prj')))
    with open(prj_name, 'w') as prj_file:
        prj_file.write(outSpatialRef.ExportToWkt())

# TEST_RAHA meters
import subprocess

def reproject_shapefile_ogr2ogr(input_shp, output_shp, target_epsg):
    """
    Reprojette un shapefile en utilisant ogr2ogr (GDAL doit être installé sur le système).
    - input_shp : chemin du shapefile source
    - output_shp : chemin du shapefile reprojeté à créer
    - target_epsg : code EPSG cible (ex : 2975)
    """
    os.environ["PROJ_LIB"] = "/usr/share/proj"  # <-- Ajoute cette ligne !
    cmd = [
        "ogr2ogr",
        "-t_srs", f"EPSG:{target_epsg}",
        output_shp,
        input_shp
    ]
    print(" ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Erreur lors de la reprojection :", result.stderr)
        raise RuntimeError("La reprojection a échoué")
    else:
        print("Reprojection réussie :", output_shp)  # FARANY



def getRasterFootprint(raster_path):
    """
    Retourne un objet ogr.Geometry représentant le polygone d’emprise du raster.
    """
    ds = gdal.Open(raster_path)
    gt = ds.GetGeoTransform()
    cols = ds.RasterXSize
    rows = ds.RasterYSize

    # Coins du raster
    points = [
        (gt[0], gt[3]),  # haut gauche
        (gt[0] + cols * gt[1], gt[3]),  # haut droite
        (gt[0] + cols * gt[1], gt[3] + rows * gt[5]),  # bas droite
        (gt[0], gt[3] + rows * gt[5]),  # bas gauche
        (gt[0], gt[3])  # retour au point de départ
    ]
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for pt in points:
        ring.AddPoint(pt[0], pt[1])
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    return poly

def createShapefileFromRasterFootprint(raster_path, output_shp, target_epsg, geom_type='polygon'):
    '''
    Crée un shapefile à partir de l’emprise d’un raster.
    - raster_path : chemin du raster source
    - output_shp : chemin du shapefile de sortie
    - target_epsg : osr.SpatialReference (projection cible)
    - geom_type : 'polygon', 'point' ou 'line'
    '''
    footprint = getRasterFootprint(raster_path)
    dic_geom = {'polygon': ogr.wkbPolygon, 'point': ogr.wkbPoint, 'line': ogr.wkbLineString}
    driver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(output_shp):
        driver.DeleteDataSource(output_shp)
    data_source = driver.CreateDataSource(output_shp)
    layer = data_source.CreateLayer('footprint', srs, dic_geom[geom_type])
    layer.CreateField(ogr.FieldDefn("Iden", ogr.OFTInteger))
    feature = ogr.Feature(layer.GetLayerDefn())
    feature.SetField("Iden", 1)
    feature.SetGeometry(footprint)
    layer.CreateFeature(feature)
    feature = None
    data_source = None


def clipShapefile(input_shp, output_shp, clip_shp):
    '''
    Coupe un shapefile (input_shp) avec un autre shapefile (clip_shp) et écrit le résultat dans output_shp.
    '''
    driver = ogr.GetDriverByName("ESRI Shapefile")
    if not os.path.exists(input_shp):
        raise FileNotFoundError(f"Shapefile d'entrée non trouvé : {input_shp}")
    if not os.path.exists(clip_shp):
        raise FileNotFoundError(f"Shapefile de clip non trouvé : {clip_shp}")
    if os.path.exists(output_shp):
        driver.DeleteDataSource(output_shp)

    inDataSource = driver.Open(input_shp, 0)
    inLayer = inDataSource.GetLayer()
    inClipSource = driver.Open(clip_shp, 0)
    inClipLayer = inClipSource.GetLayer()

    outDataSource = driver.CreateDataSource(output_shp)
    outLayer = outDataSource.CreateLayer('clip', geom_type=ogr.wkbMultiPolygon, srs=inLayer.GetSpatialRef())

    ogr.Layer.Clip(inLayer, inClipLayer, outLayer)

    # Crée le .prj
    outSpatialRef = inLayer.GetSpatialRef()
    outSpatialRef.MorphToESRI()
    file_name = os.path.basename(output_shp)
    dir_name = os.path.dirname(output_shp)
    prj_name = str(pathlib.Path(os.path.join(dir_name, file_name.split('.')[0]+'.prj')))
    with open(prj_name, 'w') as prj_file:
        prj_file.write(outSpatialRef.ExportToWkt())

    inDataSource = None
    inClipSource = None
    outDataSource = None

'''def rasterizeShapefile(input_shp, output_raster, raster_template, bc):

    #Convertit un shapefile en raster TIFF selon un raster template.
    #Si bc == '(NONE)', rasterise tout. Sinon, filtre sur le champ BEACH_CODE.

    from osgeo import ogr, gdal
    driver = ogr.GetDriverByName("ESRI Shapefile")

    #shp_ds = driver.Open(input_shp, 0)
    shp_ds = gdal.OpenEx(input_shp, gdal.OF_VECTOR)  # Utilise gdal.OpenEx pour les vecteurs
    if shp_ds is None:
        raise FileNotFoundError(f"Impossible d'ouvrir le shapefile {input_shp}")

    lyr = shp_ds.GetLayer()
    template_ds = gdal.Open(raster_template)
    if template_ds is None:
        raise FileNotFoundError(f"Impossible d'ouvrir le raster template {raster_template}")

    geot = template_ds.GetGeoTransform()
    prj = template_ds.GetProjection()
    driver = gdal.GetDriverByName('GTiff')
    new_raster_ds = driver.Create(
        output_raster, template_ds.RasterXSize, template_ds.RasterYSize, 1, gdal.GDT_Byte)
    new_raster_ds.SetGeoTransform(geot)
    new_raster_ds.SetProjection(prj)
    # filter by beach code if needed
    if bc == '(NONE)':
        gdal.RasterizeLayer(new_raster_ds, [1], lyr)
    else:
        lyr.SetAttributeFilter("BEACH_CODE IN "+bc)
        if lyr.GetFeatureCount() == 0:
            lyr.SetAttributeFilter('')
        gdal.RasterizeLayer(new_raster_ds, [1], lyr)
    new_raster_ds.GetRasterBand(1).SetNoDataValue(2)
    new_raster_data = new_raster_ds.GetRasterBand(1).ReadAsArray()
    new_raster_data[new_raster_data == 255] = 1
    new_raster_data[new_raster_data != 1] = 0
    new_raster_ds.GetRasterBand(1).WriteArray(new_raster_data)
    new_raster_ds.FlushCache()
    new_raster_ds = None
    new_raster_data = None'''

def rasterizeShapefile(input_shp, output_raster, raster_template, attribute_field, where=None):
    """Rasterize un shapefile en utilisant un raster comme template"""
    shp_ds = gdal.OpenEx(input_shp, gdal.OF_VECTOR)  # Ouvre le shapefile correctement
    if shp_ds is None:
        raise FileNotFoundError(f"Impossible d'ouvrir le shapefile {input_shp}")
    lyr = shp_ds.GetLayer()
    if lyr is None:
        raise ValueError(f"Impossible d'obtenir la couche du shapefile {input_shp}")

    # Lecture des informations du raster template
    template_ds = gdal.Open(raster_template)
    if template_ds is None:
        raise FileNotFoundError(f"Impossible d'ouvrir le raster template {raster_template}")

    geoTransform = template_ds.GetGeoTransform()
    x_res = geoTransform[1]
    y_res = -geoTransform[5]
    template_srs = template_ds.GetSpatialRef()

    # Création du raster en sortie
    target_ds = gdal.GetDriverByName('GTiff').Create(
        output_raster,
        template_ds.RasterXSize,
        template_ds.RasterYSize,
        1,
        gdal.GDT_Byte,
    )

    target_ds.SetGeoTransform(geoTransform)
    target_ds.SetProjection(template_srs.ExportToWkt())
    band = target_ds.GetRasterBand(1)
    band.SetNoDataValue(0)

    # Rasterisation
    gdal.RasterizeLayer(target_ds, [1], lyr, options=[f"ATTRIBUTE={bc}"])

    # Fermeture des datasets
    target_ds = None
    template_ds = None
    shp_ds = None


def maskPixelLine(pixel_line_path, mask_path):
    '''
    Masque le trait de côte pixelisé avec le masque des plages.
    Tous les pixels hors plage seront mis à zéro.
    '''

    if not os.path.exists(pixel_line_path):
        raise FileNotFoundError(f"Raster pixel_line non trouvé : {pixel_line_path}")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Raster mask non trouvé : {mask_path}")

    pl_ds = gdal.Open(pixel_line_path, gdal.GA_Update)
    if pl_ds is None:
        raise RuntimeError(f"Impossible d'ouvrir {pixel_line_path}")

    pl_band = pl_ds.GetRasterBand(1)
    pl_data = pl_band.ReadAsArray()

    print("pixel_line shape:", pl_data.shape)

    mask_ds = gdal.Open(mask_path)
    if mask_ds is None:
        raise RuntimeError(f"Impossible d'ouvrir {mask_path}")

    mask_band = mask_ds.GetRasterBand(1)
    mask_data = mask_band.ReadAsArray()

    print("mask shape:", mask_data.shape)

    pl_data[mask_data == 0] = 0
    pl_band.WriteArray(pl_data)
    pl_ds.FlushCache()
    pl_ds = None

def extractPoints(source_path, pl_path, processing_path, kernel, ppp, degree):
    '''
    Description:
    ------------
    Extract subpixel shoreline points based on kernel analysis over swir1 band, taking as
    template the binary mask of rough shoreline pixel line.
    Values standar used in most of the previous studies with good results are:
    3 (kernel), 4 (ppp), 3 (degree).

    for more information about this algorithm and some results:

    - "Automatic extraction of shorelines from Landsat TM and ETM+ multi-temporal images with subpixel precision". 2012.
      Remote Sensing of Environment. Josep E.Pardo-Pascual, Jaime Almonacid-Caballer, Luis A.Ruiz, Jesus Palomar-Vazquez.

    - "Assessing the Accuracy of Automatically Extracted Shorelines on Microtidal Beaches from Landsat 7,
    Landsat 8 and Sentinel-2 Imagery". 2018. Remote Sensing. Josep E. Pardo-Pascual, Elena Sanchez-Garcia, Jaime Almonacid-Caballer
    Jesus Palomar-Vazquez, Enrique Priego de los Santos, Alfonso Fernández-Sarría, Angel Balaguer-Beser.


    Arguments:
    ------------
    - source_path (string): path to the swir1 band
    - pl_path (string): path to the binary mask of rough shoreline pixel line.
    - processing_path (string): path to the folder to storage results (for each scene,
      this folder is named "temp".
    - kernel (int): kernel size in pixels. Must be an odd number
    - ppp (int): points per pixel. Number of points per pixel extracted. 4 points
      in a 20 m size resolution image means 1 point every 5 meters.
    - degree (int): degree for the mathematical fitting function. Standard values
      are 3 or 5.


    Returns:
    ------------
    - True or False (extraction was success or not)

    '''

    try:
        # opens swir1 image
        source_ds = gdal.Open(source_path)

        if source_ds is None:
            raise IOError(f"Impossible d'ouvrir {source_path}")

        source_band = source_ds.GetRasterBand(1)
        source_data = source_band.ReadAsArray()

        # opens pixel line mask image
        pl_ds = gdal.Open(pl_path)

        if pl_ds is None:
            raise IOError(f"Impossible d'ouvrir {pl_path}")

        pl_band = pl_ds.GetRasterBand(1)
        pl_data = pl_band.ReadAsArray()

        # creates output text file for coordinate points
        base_name = os.path.basename(source_path).split('.')[0]
        source_data[source_data == float('-inf')] = 0
        if os.path.isfile(str(pathlib.Path(os.path.join(processing_path, base_name+'.d')))):
            os.remove(str(pathlib.Path(os.path.join(processing_path, base_name+'.d'))))
            file_coord = open(str(pathlib.Path(os.path.join(
                processing_path, base_name+'.d'))), 'a')

        # gets swir1 features
        geoTrans = source_ds.GetGeoTransform()
        minXimage = geoTrans[0]
        maxYimage = geoTrans[3]
        dim = source_data.shape
        rows = dim[0]
        columns = dim[1]

        offset = 10  # number of rows and columns preserved to avoid overlapping in adjacent scenes

        c1 = f1 = offset
        c2 = columns - offset
        f2 = rows - offset
        resol_orig = geoTrans[1]  # pixel size
        resol = float(geoTrans[1])/ppp  # point distance
        gap = int(kernel/2)
        points_x = []
        points_y = []
        wm = computeWeights(kernel, ppp)  # weights matrix
        white_pixel = False

        for f in tqdm(range(f1, f2), desc="Extracting points"):
            for c in range(c1, c2):
                #valor = pl_data[f, c]

                #if valor == 255:  # pixel belongs to the rough pixel line
                if f < pl_data.shape[0] and c < pl_data.shape[1]:  # Vérification des limites
                    valor = pl_data[f, c]
                    if valor == 255:  # pixel belongs to the rough pixel line
                        white_pixel = True
                        nf = f
                        nc = c
                        # sub-matrix based on kernel size
                        sub = source_data[nf-gap:nf+kernel-gap, nc-gap:nc+kernel-gap]
                        # sub-matrix resampling based on ppp value
                        sub_res = rescale(sub, scale=ppp, order=3, mode='edge')
                        #cx, cy, cz = createData(sub_res, resol)  # resampled data
                        N = sub_res.shape[1] # ou shape[1] si ce n'est pas carré, mais normalement c'est carré
                        cx = np.linspace(0, (N-1)*resol, N)
                        cy = np.linspace(0, (N-1)*resol, N)
                        cz = sub_res  # cz est la matrice des valeurs

                        m = polyfit2d(cx, cy, cz, order=degree)  # fitting data

                        #order = 2  # ou le degré utilisé dans polyfit2d

                        # Transforme le vecteur en matrice :
                        m_matrix = np.array(m).reshape((degree+1, degree+1))
                        #m_matrix = np.array(m).reshape((order+1, order+1))
                        print("Taille de m :", np.array(m).size, "attendu :", (degree+1)*(degree+1))

                        # computes laplacian function
                        dx = deriva(m_matrix, 'x')
                        d2x = deriva(dx, 'x')
                        dy = deriva(m_matrix, 'y')
                        d2y = deriva(dy, 'y')
                        laplaciano = d2x+d2y

                        cxg, cyg = np.meshgrid(cx, cy)


                        laplaciano_eval = polyval2d(cxg, cyg, laplaciano, order=degree)

                        print(cx.shape, cy.shape, laplaciano_eval.shape)

                        # get contour points for laplacian = 0
                        v = verticeslaplaciano(cx, cy, laplaciano_eval, kernel, ppp)
                        #v = verticeslaplaciano(cxg, cyg, laplaciano, kernel, ppp)


                        if v != None:
                            if len(v) != 0:
                                # if there are more than one contour, we select the contour with highest slope and more centered
                                indice = mejor_curva_pendiente3(v, dx, dy, wm, resol)
                                if indice != -1:
                                    linea = v[indice]
                                    for i in range(0, len(linea)):
                                        par = linea[i]
                                        points_x.append(par[0])
                                        points_y.append(par[1])

                                        # writes the contour points to the text file
                                        escribeCoords(points_x, points_y, minXimage, maxYimage,
                                          resol_orig, resol, wm, nf, nc, kernel, file_coord)
                                    points_x = []
                                    points_y = []
                        file_coord.close()

        if white_pixel:
            # variable release
            del source_ds
            del source_band
            del source_data
            del pl_data
            del sub
            del sub_res
            del m
            del cx
            del cy
            del cz
            del wm
            gc.collect()
            return True

    except Exception as e:
        print(f"Erreur dans extractPoints: {e}")
        return False



def computeWeights(kernel, ppp):
    '''
    Description:
    ------------
    Computes a matrix with values that follows a normal distribution.
    It is used to ponderate the extracted points based on the distance
    of each point to the center of the image kernel

    Arguments:
    ------------
    - kernel (int): kernel size in pixels. Must be an odd number
    - ppp (int): points per pixel. Number of points per pixel extracted.

    Returns:
    ------------
    - p (numpy matrix): weights matrix

    '''

    p = np.zeros((kernel*ppp, kernel*ppp))
    f, c = p.shape
    cont_i = cont_j = 1.0
    for i in range(0, f):
        for j in range(0, c):
            d = np.sqrt((cont_i-(float(f)+1.0)/2.0)**2 +
                        (cont_j-(float(c)+1.0)/2.0)**2)
            p[i, j] = normcdf(-d, 0, 3)*2
            cont_j += 1
        cont_i += 1
        cont_j = 1
    return p


def createData(image, resol):
    '''
    Description:
    ------------
    Creates x, y, z arrays of the resampled kernel

    Arguments:
    ------------
    - image (numpy matrix): resampled kernel
    - resol (float): swir1 spatial resolution

    Returns:
    ------------
    - z, y, z (float arrays)

    '''
    inicio = resol-(resol/2.0)  # pixel center
    z = (np.ravel(image)).astype(float)
    tamdata = int(np.sqrt(len(z)))
    x, y = np.meshgrid(np.arange(inicio, tamdata*resol, resol),
                       np.arange(inicio, tamdata*resol, resol))
    x = (np.ravel(x)).astype(float)
    y = (np.ravel(y)).astype(float)
    return x, y, z


def verticeslaplaciano(x, y, m, kernel, ppp):
    '''
    Description:
    ------------
    Computes contour points from laplacian function = 0.
    Uses matplotlib contour function

    Arguments:
    ------------
    - x, y, m (numpy 1D arrays): x, y , z coordinates for laplacian function
    - axis (string): axis to compute the derivative function
    - kernel (int): kernel size in pixels. Must be an odd number
    - ppp (int): points per pixel. Number of points per pixel extracted.

    ze in pixels. Must be an odd number
    - ppp (int): points per p

    Returns:
    ------------
    - v (list): list of contour vertices

    '''
    clf()
    v = []
    #zz = polyval2d(x, y, m)
    #x = np.reshape(x, (kernel*ppp, kernel*ppp))
    #y = np.reshape(y, (kernel*ppp, kernel*ppp))
    #zz = np.reshape(zz, (kernel*ppp, kernel*ppp))

    # Crée la grille 2D
    xg, yg = np.meshgrid(x, y)

    # Applique le polynôme sur la grille
    #zz = polyval2d(xg, yg, m, order=2)
    #zz = m  # m est déjà le laplacien évalué sur la grille

    try:  # Prevents errors in contour computing
        CS = contour(x, y, m, 0, colors='y')
        curvas = get_contour_verts(CS)
        for curva in curvas:
            for parte in curva:
                v.append(parte)
        return v
    except Exception as e:
        print(f"Erreur dans contour : {e}")
        return None


def get_contour_verts(cn):
    '''
    Description:
    ------------
    Extract vertices from a contour

    Arguments:
    ------------
    - cn (object): matplotlib contour object

    Returns:
    ------------
    - contours (list): list of contours vertices

    '''

    contours = []
    # for each contour line
    for cc in cn.collections:
        paths = []
        # for each separate section of the contour line
        for pp in cc.get_paths():
            xy = []
            # for each segment of that section
            for vv in pp.iter_segments():
                xy.append(vv[0])
            paths.append(np.vstack(xy))
        contours.append(paths)
    return contours



def mejor_curva_pendiente3(v, dx, dy, mp, resol):
    '''
    Description:
    ------------
    Select best contour based on the highest mean slope and centrality criteria

    Arguments:
    ------------
    - v (list): list of contours
    - dx (numpy matrix): first derivative of the fitting function in X axis (slope)
    - dy (numpy matrix): first derivative of the fitting function in Y axis (slope)
    - mp (numpy matrix): weight matrix (centrality criteria)

    Returns:
    ------------
    - candidate (int): index of the selected contour

    '''
    pendientes = []
    p_max = 0
    candidate = -1
    for i, curva in enumerate(v):
        for par in curva:
            x = par[0]
            y = par[1]
            #px = abs(polyval2d([x], [y], dx))
            #py = abs(polyval2d([x], [y], dy))
            px = abs(polyval2d(np.array([x]), np.array([y]), dx)[0]) # Correction ici
            py = abs(polyval2d(np.array([x]), np.array([y]), dy)[0]) # Correction ici
            p = np.sqrt(px**2+py**2)
            peso = mp[int(x/resol), int(y/resol)]
            pendientes.append(p*peso)
        p_med = np.average(pendientes)
        if p_med >= p_max:
            p_max = p_med
            candidate = i
        pendientes = []
    return candidate


def escribeCoords(x, y, xmin, ymax, resol_orig, resol, wm, fil, col, kernel, output_file):
    '''
    Description:
    ------------
    Write extracted contours vertices coordinates to the .txt file.
    The original points are in subpixel image coordinates. They have to be
    converted to world coordinates

    Arguments:
    ------------
    - x (list): list of X coordinates
    - y (list): list of Y coordinates
    - xmin (float): minimum X coordinate of the swir1 image
    - ymin (float): minimum Y coordinate of the swir1 image
    - resol_orig (float): spatial resolution of the swir1 image
    - resol (float): map distance among each extracted point (pixel size / ppp)
    - wm (numpy matrix): weight matrix
    - fil (int): row coordinate for the center pixel in the current kernel
    - col (int): column coordinate for the center pixel in the current kernel
    - kernel (int): kernel size in pixels
    - output_file (string): path to the output file

    Returns:
    ------------
    None

    '''

    for i in range(0, len(x)):
        # coordenadas punto sobre imagen global
        rx = xmin+(col-int(kernel/2.0))*resol_orig+x[i]
        ry = ymax-(fil-int(kernel/2.0))*resol_orig-y[i]
        peso = wm[int(x[i]/resol), int(y[i]/resol)]
        output_file.write(str(rx)+","+str(ry)+","+str(peso)+'\n')



def deriva(m, axis):
    '''
    Description:
    ------------
    Computes derivative function of a matrix in a particular axis

    Arguments:
    ------------
    - m (numpy matrix): input matrix
    - axis (string): axis to compute the derivative function

    Returns:
    ------------
    - nm (numpy array): derivative function

    '''

    f, c = m.shape
    if axis == 'x':
        factores = range(1, c)
        nm = m[:, range(1, c)]
        nm = nm*factores
        ceros = np.zeros((f,), dtype=float)
        nm = np.vstack((nm.T, ceros.T)).T
        return nm

    if axis == 'y':
        factores = range(1, f)
        nm = m[range(1, f), :]
        nm = (nm.T*factores).T
        ceros = np.zeros((c,), dtype=float)
        nm = np.vstack((nm, ceros))
        return nm


def normcdf(x, mu, sigma):
    '''
    Description:
    ------------
    Computes the normal distribution value

    Arguments:
    ------------
    - x (float): distance from the center of kernel
    - mu: mean of the normal distribution
    - sigma: standar deviation of the normal distribution

    Returns:
    ------------
    - y (float): normal distribution value

    '''

    t = x-mu
    y = 0.5*erfcc(-t/(sigma*sqrt(2.0)))
    if y > 1.0:
        y = 1.0
    return y



def erfcc(x):
    """Complementary error function."""
    z = abs(x)
    t = 1. / (1. + 0.5*z)
    r = t * exp(-z*z-1.26551223+t*(1.00002368+t*(.37409196 +
                                                 t*(.09678418+t*(-.18628806+t*(.27886807 +
                                                                               t*(-1.13520398+t*(1.48851587+t*(-.82215223 +
                                                                                                               t*.17087277)))))))))
    if (x >= 0.):
        return r
    else:
        return 2. - r


# *******************************************************************************
# SECTION: AVERAGE POINTS FUNCTIONS
# *******************************************************************************


def averagePoints(source_path, processing_path, cluster_distance, min_cluster_size):
    '''
    Description:
    ------------
    Takes the subpixel rough extracted points and computes the average points.
    The algorithm scan in X and Y direction points with the same coordinates and
    makes groups of points by using clustering criteria as maximum distance among
    points and minimum number of points in a cluster.
    Creates a .txt file with the results

    Arguments:
    ------------
    - source_path (string): path to the scene
    - processing_path (string): path to the processing folder
    - cluster_distance (int): maximum distance between two points to be consider a cluster
    - min_cluster_size (int): minimum number of points in a cluster

    Returns:
    ------------
    None

    '''

    # reading coordinates from extracted subpixel points in the kernel analysis
    base_name = os.path.basename(source_path).split('.')[0]
    file_name = str(pathlib.Path(
        os.path.join(processing_path, base_name+'.d')))
    with open(file_name, 'r') as fichero:
        iter1 = csv.reader(fichero, delimiter=',')
        datos = np.asarray([[dato[0], dato[1]]
                           for dato in iter1]).astype(float)
        fichero.seek(0)
        iter2 = csv.reader(fichero, delimiter=',')
        pesos = np.asarray([dato[2] for dato in iter2]).astype(float)

    ejex = np.unique(datos[:, 0])  # unique values on the x-axis
    ejey = np.unique(datos[:, 1])  # unique values on the y-axis

    # computing clusters
    medias = creaCluster(datos, pesos, ejex, ejey,
                         cluster_distance, min_cluster_size)

    # writes results to the output file (average x and average y of every cluster)
    with open(str(pathlib.Path(os.path.join(processing_path, base_name+'.m'))), 'w') as fichero:
        for media in medias:
            fichero.write(str(media[0])+","+str(media[1])+"\n")


def creaCluster(d, p, ex, ey, cluster_distance, min_cluster_size):
    '''
    Description:
    ------------
    Makes groups of points acording clustering criteria (maximum distance among
    points and minimum number of points in a cluster). From each group, the algorithm
    computes a ponderate average value for x and y coordinates based on a weight matrix.

    Arguments:
    ------------
    - d (numpy array): list of X-Y coordinates
    - p (numpy matrix): weight matrix
    - ex (numpy array): unique values on the x-axis
    - ey (numpy array): unique values on the y-axis
    - cluster_distance (int): maximum distance between two points to be consider a cluster
    - min_cluster_size (int): minimum number of points in a cluster

    Returns:
    ------------
    - average_points (list): list of average points for each cluster

    '''

    tol = cluster_distance
    average_points = []

    # clustering in x-axis
    for x in ex:
        id_x = np.nonzero(d[:, 0] == x)
        cy = d[:, 1][id_x]
        pey = p[id_x]
        if len(cy) >= 2:
            orig_coord, pos = getClusters(cy, tol)
            for cp in pos:
                cluster = orig_coord[cp]
                if len(cluster) >= min_cluster_size:
                    p_cluster = pey[cp]
                    media_y = np.average(cluster, weights=p_cluster)
                    average_points.append([x, media_y])

    # clustering in y-axis
    for y in ey:
        id_y = np.nonzero(d[:, 1] == y)
        cx = d[:, 0][id_y]
        pex = p[id_y]
        if len(cx) >= 2:
            orig_coord, pos = getClusters(cx, tol)
            for cp in pos:
                cluster = orig_coord[cp]
                if len(cluster) >= min_cluster_size:
                    p_cluster = pex[cp]
                    media_x = np.average(cluster, weights=p_cluster)
                    average_points.append([media_x, y])
    return average_points


def getClusters(coord, tol):
    '''
    Description:
    ------------
    Makes groups of points based on a maximum distance.

    Arguments:
    ------------
    - coord (list): list of point coordinates with the same x or y value
    - tol (int): cluster distance (maximum distance between two points to
      be consider a cluster)

    Returns:
    ------------
    - orig_coord (list): list of point coordinates with the same x or y value
    - pos (list): index of points that belong tho the same cluster

    '''

    clusters = []
    cluster = []
    orig_coord = coord.copy()
    coord.sort()
    cluster.append(0)
    for i in range(0, len(coord)-1):
        current = coord[i]
        siguiente = coord[i+1]
        dist = siguiente-current
        if dist <= tol:
            cluster.append(i+1)
        else:
            clusters.append(cluster)
            cluster = []
            cluster.append(i+1)
    clusters.append(cluster)
    parcial = []
    pos = []
    for c in clusters:
        for iden in c:
            a, = np.where(orig_coord == coord[iden])
            parcial.append(a[0])
        pos.append(parcial)
        parcial = []
    return orig_coord, pos

# *******************************************************************************
# SECTION: SHAPEFILE CREATION FUNCTIONS
# *******************************************************************************


def createShpFromAverageFile(source_path, processing_path):
    '''
    Description:
    ------------
    Converts average point coordinates from .txt file to .shp file.
    The name of the shapefile is based on the name of the .xtx file.
    In order to copy the beach code to each average point, an attribute
    field call "BEACH_CODE" is added.

    Arguments:
    ------------
    - source_path (string): path to the template swir1 image
    - processing_path (string): path to the processing output path.

    Returns:
    ------------
    - shp_path (string): path to the .shp file

    '''

    # gets projection from template image
    source_ds = gdal.Open(source_path)
    prj = osr.SpatialReference()
    prj.ImportFromWkt(source_ds.GetProjectionRef())

    # reads average point coordinates from .txt file
    base_name = os.path.basename(source_path).split('.')[0]
    file_name = str(pathlib.Path(
        os.path.join(processing_path, base_name+'.m')))
    with open(file_name, 'r') as f:
        data = csv.reader(f, delimiter=',')
        coords = np.asarray([[dato[0], dato[1]]
                            for dato in data]).astype(float)

    # creates a new shapefile and adds a database structure
    driver = ogr.GetDriverByName("ESRI Shapefile")
    shp_path = str(pathlib.Path(os.path.join(
        processing_path, base_name+'.shp')))
    if os.path.exists(shp_path):
        driver.DeleteDataSource(shp_path)
    data_source = driver.CreateDataSource(shp_path)
    layer = data_source.CreateLayer(' ', geom_type=ogr.wkbPoint, srs=prj)
    id_field = ogr.FieldDefn("Id_pnt", ogr.OFTInteger)
    id_field.SetWidth(10)
    layer.CreateField(id_field)
    id_field2 = ogr.FieldDefn("BEACH_CODE", ogr.OFTInteger)
    id_field2.SetWidth(10)
    layer.CreateField(id_field2)
    layer_defn = layer.GetLayerDefn()

    # creates shapefile features
    for i in range(0, len(coords)):
        values = coords[i]
        feat = ogr.Feature(layer_defn)
        pnt = ogr.Geometry(ogr.wkbPoint)
        pnt.AddPoint(values[0], values[1])
        feat.SetGeometry(pnt)
        feat.SetField('Id_pnt', i)
        feat.SetField('BEACH_CODE', 0)
        layer.CreateFeature(feat)
    data_source.FlushCache()
    source_ds = None
    return shp_path

def envelopeToGeom(geom):
    '''
    Description:
    ------------
    Returns the bounding box of a polygon geometry.

    Arguments:
    ------------
    - geom (objetc): ogr geometry object

    Returns:
    ------------
    poly_envelope (object): ogr geometry object

    '''

    # gets bounding box from geometry
    (minX, maxX, minY, maxY) = geom.GetEnvelope()

    # creates ring
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(minX, minY)
    ring.AddPoint(maxX, minY)
    ring.AddPoint(maxX, maxY)
    ring.AddPoint(minX, maxY)
    ring.AddPoint(minX, minY)

    # creates polygon
    poly_envelope = ogr.Geometry(ogr.wkbPolygon)
    poly_envelope.AddGeometry(ring)
    return poly_envelope

def addIdField(input_path):
    '''
    Description:
    ------------
    Adds the field "ID_FEAT" to the input shapefile

    Arguments:
    ------------
    - input_path (string): path to the input shapefile

    Returns:
    ------------
    None

    '''
    # open shapefile
    driver = ogr.GetDriverByName('ESRI Shapefile')
    ds_shp = driver.Open(input_path, 1)
    layer_shp = ds_shp.GetLayer()

    layer_shp_defn = layer_shp.GetLayerDefn()
    field_names = [layer_shp_defn.GetFieldDefn(
        i).GetName() for i in range(layer_shp_defn.GetFieldCount())]

    # adds id field
    if not 'ID_FEAT' in field_names:  # to ensure that attribute "ID_FEAT" exists
        new_field = ogr.FieldDefn('ID_FEAT', ogr.OFTInteger)
        layer_shp.CreateField(new_field)

    # populates id field
    id_feat_shp = 0
    for feat_shp in layer_shp:
        feat_shp.SetField('ID_FEAT', id_feat_shp)
        layer_shp.SetFeature(feat_shp)
        id_feat_shp += 1

    ds_shp.FlushCache()
    ds_shp = None


def exportToGeojson(shp_path):
    '''
    Description:
    ------------
    Exports shapefile to GeoJson format

    Arguments:
    ------------
    - shp_path (string): path to the input shapefile

    Returns:
    ------------
    None

    '''
    base_name = os.path.basename(shp_path).split('.')[0]
    dir_name = os.path.dirname(shp_path)
    geojson_path = str(pathlib.Path(
        os.path.join(dir_name, base_name+'.json')))
    with shapefile.Reader(shp_path) as shp:
        geojson_data = shp.__geo_interface__
        with open(geojson_path, 'w') as geojson_file:
            geojson_file.write(json.dumps(geojson_data))


def copyShpIdentifiers(shp_polygons, shp_points):
    '''
    Description:
    ------------
    Copies the beach code from the beach shapefile to each average point using
    two geometry intersection test.

    Arguments:
    ------------
    - shp_polygons (string): path to beaches shapefile
    - shp_points (string): path to the points shapefile.

    Returns:
    ------------
    None

    '''

    # opens both polygons and points shapefiles
    driver = ogr.GetDriverByName('ESRI Shapefile')
    ds_pol = driver.Open(shp_polygons, 0)
    ds_point = driver.Open(shp_points, 1)

    layer_pol = ds_pol.GetLayer()
    layer_point = ds_point.GetLayer()

    layer_point_defn = layer_point.GetLayerDefn()
    field_names = [layer_point_defn.GetFieldDefn(
        i).GetName() for i in range(layer_point_defn.GetFieldCount())]

    # adds beach code field
    if not 'BEACH_CODE' in field_names:  # to ensure that attribute "BEACH_CODE" exists
        new_field = ogr.FieldDefn('BEACH_CODE', ogr.OFTInteger)
        layer_point.CreateField(new_field)

    # populates beach code field
    for feat_pol in layer_pol:
        id_feat_pol = feat_pol.GetField('BEACH_CODE')
        geom_pol = feat_pol.GetGeometryRef()
        geom_envelope = envelopeToGeom(geom_pol)
        for feat_point in layer_point:
            geom_point = feat_point.GetGeometryRef()
            if geom_point.Intersect(geom_envelope):  # first intersection test
                if geom_point.Intersect(geom_pol):  # second intersection test
                    feat_point.SetField('BEACH_CODE', id_feat_pol)
                    layer_point.SetFeature(feat_point)

    ds_point.FlushCache()
    ds_point = None
    ds_pol = None


# *******************************************************************************
# SECTION: POINT CLEANING FUNCTIONS
# *******************************************************************************

def cleanPoints2(shp_path, tol_rba, level):
    '''
    Description:
    ------------
    Remove outliers points based on two criteria:
    - longest spanning tree algorithm (LST).
    - angle tolerance.

    To improve the performance, the algorithm uses an initial Delaunay triangulation
    to create a direct graph.

    Two versions of cleaned points shapefile is created: point and line versions

    More information:
    "An efficient protocol for accurate and massive shoreline definition from
    mid-resolution satellite imagery". 2020. Coastal Engineering. E. Sanchez-García,
    J.M. Palomar-Vazquez, J.E. Pardo-Pascual, J. Almonacid-Caballer, C. Cabezas-Rabadan,
    L. Gomez-Pujol.

    Arguments:
    ------------
    - shp_path (string): path to the points shapefile
    - tol_rba (int): angle tolerance
    - level (int): takes one point every n (level) points. Speeds the process

    Returns:
    ------------
    None

    '''
    # opens the shapefile
    driver = ogr.GetDriverByName("ESRI Shapefile")
    source_ds = driver.Open(shp_path, 0)
    prj = source_ds.GetLayer().GetSpatialRef()
    base_name = os.path.basename(shp_path).split('.')[0]
    dir_name = os.path.dirname(shp_path)
    layer = source_ds.GetLayer()

    # gest list of unique BEACH_CODE values
    ids = []
    for feature in layer:
        id_feat = feature.GetField('BEACH_CODE')
        if not id_feat is None:
            ids.append(id_feat)
    ids = list(set(ids))
    layer.ResetReading()

    ids.sort()
    # prevents from points with BEACH_CODE 0 (outside of any beach area).
    if ids[0] == 0:
        ids.remove(0)  # removes points

    # creates groups of points with the same BEACH_CODE value
    groups = []
    identifiers = []
    for id_feat in ids:
        geometries = []
        layer.SetAttributeFilter("BEACH_CODE = "+str(id_feat))
        for feature in layer:
            geom = feature.GetGeometryRef()
            if not geom is None:
                geometries.append(geom.Clone())
        groups.append(geometries)
        identifiers.append(id_feat)

    # process each group separately
    clean_geometries = []
    level = 1
    for i in range(0, len(groups)):
        group = groups[i]
        identifier = identifiers[i]
        coords = []
        ng = float(len(group))
        # prevents from too much long numer of points in a group
        # level = ceil(ng/group_size)
        for i in range(0, len(group), level):
            geom = group[i].Clone()
            coords.append([geom.GetX(), geom.GetY()])
        points = np.array(coords)
        if len(points >= 4):  # delaunay triangulation needs 4 or more points
            try:
                tri = Delaunay(points)
                # list of triangles
                lista_tri = tri.simplices
                # list of ids of the connected points wiht LST
                lst = computeLST(lista_tri, points)
                # remove LST point by angle tolerance
                clean_points = cleanPointsByAngle(lst, points, tol_rba)
                # list of cleaned points with its identifier
                clean_geometries.append([clean_points, identifier])
            except:
                pass

    # crates point and line versions of the cleaned points
    makePointShp(str(pathlib.Path(os.path.join(dir_name, base_name +
                 '_cp.shp'))), clean_geometries, prj)
    makeLineShp(str(pathlib.Path(os.path.join(dir_name, base_name +
                '_cl.shp'))), clean_geometries, prj)
    source_ds = None
###############################################################################################################################################################################################################################

# FONCTION PRINCIPALE

###############################################################################################################################################################################################################################



# ------------------------------ Définir la projection cible (WGS 84) ------------------------------

target_epsg = 4326

# ------------------------------ Affichage de l'évolution de chaque traitement ------------------------------ '

total_steps = 12
current_step = 1

# Découpe chaque bande sur cette grille de référence
beach_shp = "/home/jonathan/SAET/SAET_installation/Reunion_Island_Boundary.shp"

#beaches_shp = "/home/jonathan/SAET/SAET_installation/beaches_Reunion_Island2.shp" # /home/jonathan/SAET/SAET_installation/aux_data/

#shp_path = "/home/jonathan/SAET/SAET_installation/beaches_Reunion_Island2.shp"


if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for date_folder in os.listdir(root_path):
    scene_path = os.path.join(root_path, date_folder)

    if not os.path.isdir(scene_path):
        continue

    # Création des dossiers temp et SDS pour cette image
    temp_dir = os.path.join(scene_path, "Temp") # ----------------------------- Dossier temporaire pour les sous enregistrement des étapes intérmédiaires -----------------------------
    os.makedirs(temp_dir, exist_ok=True)
    sds_dir = os.path.join(scene_path, "SDS") # ----------------------------- Dossier qui contient les résultatà la fin -----------------------------
    createFolderCheck(temp_dir)
    createFolderCheck(sds_dir)

    print(f"Traitement du dossier: {date_folder}")

    # ----------------------------- Recherche des bandes avec la bonne nomenclature (*.jp2) -----------------------------

    try:
        # ----------------------------- Recherche adaptée à la nomenclature T40KCB_20240709T063509_B8_20m.jp2 -----------------------------

        band_paths = {
            'B02': glob.glob(os.path.join(scene_path, "*_B02_*.jp2"))[0],  # Blue
            'B03': glob.glob(os.path.join(scene_path, "*_B03_*.jp2"))[0],  # Green
            'B04': glob.glob(os.path.join(scene_path, "*_B04_*.jp2"))[0],  # Red
            'B08': glob.glob(os.path.join(scene_path, "*_B08_*.jp2"))[0],  # NIR
            'B11': glob.glob(os.path.join(scene_path, "*_B11_*.jp2"))[0],  # SWIR1
            'B12': glob.glob(os.path.join(scene_path, "*_B12_*.jp2"))[0],  # SWIR2
        }
    except IndexError:

        # ----------------------------- Essai avec une nomenclature alternative -----------------------------
        try:
            band_paths = {
                'B02': glob.glob(os.path.join(scene_path, "*_B02_*.jp2"))[0],  # Blue
                'B03': glob.glob(os.path.join(scene_path, "*B03*.jp2"))[0],  # Green
                'B04': glob.glob(os.path.join(scene_path, "*B04*.jp2"))[0],  # Red
                'B08': glob.glob(os.path.join(scene_path, "*B08*.jp2"))[0],  # NIR
                'B11': glob.glob(os.path.join(scene_path, "*B11*.jp2"))[0],  # SWIR1
                'B12': glob.glob(os.path.join(scene_path, "*B12*.jp2"))[0],  # SWIR2
            }
        except IndexError:
            print(f"Impossible de trouver les bandes dans {scene_path}")
            print(f"Contenu du dossier: {os.listdir(scene_path)}")
            continue

    print(f"Bandes trouvées: {band_paths}")

    # ----------------------------- Reprojection EPSG:2975 -----------------------------
    projected_band_paths = {}
    for band_key, band_path in band_paths.items():
        out_proj_path = os.path.join(temp_dir, f"{date_folder}_{band_key}_4326.tif")
        reprojectRaster(band_path, out_proj_path, target_epsg)
        projected_band_paths[band_key] = out_proj_path

    # === Ici, place le calcul de la grille de référence ===
    ref_band_path = projected_band_paths['B11']
    ref_ds = gdal.Open(ref_band_path)
    ref_geo = ref_ds.GetGeoTransform()
    ref_proj = ref_ds.GetProjection()
    ref_xsize = ref_ds.RasterXSize
    ref_ysize = ref_ds.RasterYSize
    ref_bounds = (
        ref_geo[0],
        ref_geo[3] + ref_ysize * ref_geo[5],
        ref_geo[0] + ref_xsize * ref_geo[1],
        ref_geo[3]
    )

    masked_band_paths = {}
    all_ok = True
    for band_key, band_path in projected_band_paths.items():
        masked_path = os.path.join(temp_dir, f"{date_folder}_{band_key}_masked.tif")
        mask_raster_with_shapefile(band_path, beach_shp, masked_path,
                                   ref_bounds, ref_xsize, ref_ysize, ref_proj)

        if not os.path.exists(masked_path):
            print(f"Erreur : le fichier {masked_path} n'a pas été généré. Vérifiez la validité du shapefile utilisé comme masque.")
            all_ok = False
            break
        masked_band_paths[band_key] = masked_path

    if not all_ok or len(masked_band_paths) != 6:
        print("Erreur : toutes les bandes masquées n'ont pas été générées, arrêt du traitement pour cette scène.")
        continue

    # 1. Générer le masque de nuages
    cloud_mask_path = createCloudMaskS2(scene_path, output_dir, cloud_mask_level='2') # masquage le plus complet et le plus sûr : les ombres de nuages (SCL=3) ; les nuages moyens (SCL=8) ; les nuages denses (SCL=9) ; les cirrus (SCL=10). Peut choisir 0: Aucun nuage n’est masqué, 1: Seuls les nuages denses (SCL=9) sont masqués.

    if cloud_mask_path is None or not os.path.exists(cloud_mask_path):
        print("Erreur : le masque de nuages n'a pas été généré.")
        continue

    # Récupère le nom de fichier sans extension
    basename = os.path.basename(cloud_mask_path)
    name, ext = os.path.splitext(basename)

    cloud_mask_resampled_path = os.path.join(temp_dir, f"{name}_resampled{ext}")

    resample_raster_to_match(cloud_mask_path, masked_band_paths['B02'],
                             cloud_mask_resampled_path)
    cloud_mask = getBandData(cloud_mask_resampled_path)
    print("cloud_mask.shape après resampling :", cloud_mask.shape)

    # Charger les bandes
    data_blue  = getBandData20m(masked_band_paths['B02'])   # Blue
    data_green = getBandData20m(masked_band_paths['B03'])   # Green
    data_red   = getBandData20m(masked_band_paths['B04'])   # Red
    data_nir   = getBandData20m(masked_band_paths['B08'])   # NIR
    data_swir1 = getBandData20m(masked_band_paths['B11'])   # SWIR1
    data_swir2 = getBandData20m(masked_band_paths['B12'])   # SWIR2

    # Appliquer le masque de nuages à chaque bande, si les tailles correspondent
    band_arrays = [data_blue, data_green, data_red, data_nir, data_swir1, data_swir2]

    band_names = ['data_blue', 'data_green', 'data_red', 'data_nir', 'data_swir1', 'data_swir2']

    for name, arr in zip(band_names, band_arrays):
        if arr.shape == cloud_mask.shape:
            arr[cloud_mask == 2] = np.nan
        else:
            print(f"Erreur : dimensions incompatibles pour le masquage nuage sur {name} ({arr.shape} vs {cloud_mask.shape})")

    # Sauvegarder les rasters d'indices
    template_band = band_paths['B11']

    # Calcul des indices (adapter selon tes besoins)
    show_progress(current_step, total_steps, "Calcul MNDWI, MNDWI_mask, MNDWI_tif, MNDWI_mask_tif")
    mndwi, mndwi_mask, mndwi_tif, mndwi_mask_tif = process_mndwi(data_green, data_swir1, temp_dir, date_folder, template_band)
    current_step += 1

    show_progress(current_step, total_steps, "Calcul AWEINSH, AWEINSH_mask, AWEINSH_tif, AWEINSH_mask_tif")
    aweinsh, aweinsh_mask, aweinsh_tif, aweinsh_mask_tif = process_aweinsh(data_green, data_nir, data_swir1, data_swir2, date_folder, template_band)
    current_step += 1

    show_progress(current_step, total_steps, "Calcul AWEISH, AWEISH_mask, AWEISH_tif, AWEISH_mask_tif")
    aweish, aweish_mask, aweish_tif, aweish_mask_tif = process_aweish(data_blue, data_green, data_nir, data_swir1, data_swir2, date_folder, template_band)
    current_step += 1

    show_progress(current_step, total_steps, "Calcul NDWI, NDWI, NDWI_mask, NDWI_tif, NDWI_mask_tif")
    ndwi, ndwi_mask, ndwi_tif, ndwi_mask_tif = process_ndwi(data_green, data_nir, temp_dir, date_folder, template_band)
    current_step += 1

    show_progress(current_step, total_steps, "Calcul NDVI, NDVI_mask, NDVI_tif, NDVI_mask_tif")
    ndvi, ndvi_mask, ndvi_tif, ndvi_mask_tif = process_ndvi(data_nir, data_red, temp_dir, date_folder, template_band)
    current_step += 1

    show_progress(current_step, total_steps, "Calcul IB, IB_mask, IB_tif, IB_mask_tif")
    IB, IB_mask, IB_tif, IB_mask_tif = process_ib(data_red, data_nir, temp_dir, date_folder, template_band)
    current_step += 1

    # === Combinaison pondérée des masques d’indices ===
    show_progress(current_step, total_steps, "Combinaison pondérée pour avoir le pixel d'eau")
    combination_index, combination_mask, comb_tif, comb_tif_mask = process_combination_mask(
        mndwi_mask, aweish_mask, aweinsh_mask, ndwi_mask, ndvi_mask, IB_mask,
        temp_dir, date_folder, template_band,
        weights=(7, 6, 5, 4, 2, 1),
        threshold=10
        )
    current_step += 1

    # Resample le masque nuage sur la grille du masque eau/terre (comb_tif_mask ou mask_path)
    cloud_mask_resampled_path = os.path.splitext(cloud_mask_path)[0] + "_resampled.tif"
    resample_raster_to_match(cloud_mask_path, comb_tif_mask, cloud_mask_resampled_path)

    # Extraction du trait de côte pixelisé
    show_progress(current_step, total_steps, "Extraction du trait de côte pixelisé")
    pixel_line = createPixelLine(
        method='erosion',
        mask_path=comb_tif_mask,
        cloud_mask_path=cloud_mask_resampled_path,
        cloud_buffer=9
        )
    pixel_line_tif = os.path.join(temp_dir, f"{date_folder}_pixel_line.tif")
    saveIndex(pixel_line, pixel_line_tif, template_band, dType=gdal.GDT_Byte)
    current_step += 1

    #logging.info('Début de la reprojection...')
    show_progress(current_step, total_steps, "Début de la reprojection...")

    from osgeo import ogr, gdal

    #print("GDAL version:", gdal.VersionInfo())

    beaches_shp = "/home/jonathan/SAET/SAET_installation/beaches_Reunion_Island.shp"

    ds = ogr.Open(beaches_shp)

    '''if ds is None:
        print("Erreur d’ouverture du shapefile (avant reprojection) !")
        exit(1)
    else:
        print("Shapefile ouvert avec succès.")'''

    from osgeo import ogr, osr
    drv = ogr.GetDriverByName("ESRI Shapefile")
    test_shp = os.path.join(temp_dir, "test.shp")
    if os.path.exists(test_shp):
        drv.DeleteDataSource(test_shp)


    '''# TEST_P
    ds = drv.CreateDataSource(test_shp)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    layer = ds.CreateLayer("test", srs=srs)
    ds = None
    print("Shapefile test créé ?")
    print(os.path.isfile(test_shp)) # TEST_P'''

    source_epsg = getSourceEpsg(beaches_shp) # int, ex: 4326
    target_epsg = getTargetEpsg(scene_path, 'B11') # int, ex: 2975

    source_srs = osr.SpatialReference()
    source_srs.ImportFromEPSG(source_epsg)
    target_srs = osr.SpatialReference()
    target_srs.ImportFromEPSG(target_epsg)

    #beach_shp_reproj = os.path.join(temp_dir, f"{date_folder}_bb300_r.shp")
    #reprojectShp(beaches_shp, beach_shp_reproj, source_srs, target_srs)

    # Génère le nom de sortie
    output_shp = os.path.join(temp_dir, f"{date_folder}_bb300_r.shp")

    # Appelle la fonction de reprojection
    reproject_shapefile_ogr2ogr(beaches_shp, output_shp, target_epsg)

    current_step += 1

    show_progress(current_step, total_steps, "Computing footprint band...")

    #print("Fichiers présents dans Temp :", os.listdir(temp_dir))

    # Détermination de la projection cible à partir de la bande B11
    b11_path = band_paths['B11']
    ds = gdal.Open(b11_path)
    proj_wkt = ds.GetProjection()
    srs = osr.SpatialReference()
    srs.ImportFromWkt(proj_wkt)


    # Création du shapefile d’emprise
    #logging.info('Computing footprint band...')
    '''footprint_shp = str(pathlib.Path(os.path.join(temp_dir,
                                                  f"{date_folder}_scene_footprint.shp")))
    createShapefileFromRasterFootprint(b11_path, footprint_shp, srs,
                                       geom_type='polygon')'''

    footprint_shp = os.path.join(temp_dir, f"{date_folder}_scene_footprint.shp")
    createShapefileFromRasterFootprint(b11_path, footprint_shp, srs, geom_type='polygon')

    current_step += 1

    # ETO IZAO

    show_progress(current_step, total_steps, "Clipping shp of beaches by scene footprint...")

    #logging.info('Clipping shp of beaches by scene footprint...')
    #input_shp = str(pathlib.Path(os.path.join(temp_dir, 'bb300_r.shp')))
    #output_shp = str(pathlib.Path(os.path.join(temp_dir, 'clip_bb300_r.shp')))
    #clip_shp = str(pathlib.Path(os.path.join(temp_dir, 'scene_footprint.shp')))
    input_shp = os.path.join(temp_dir, f"{date_folder}_bb300_r.shp")
    output_shp = os.path.join(temp_dir, f"{date_folder}_clip_bb300_r.shp")
    clip_shp = footprint_shp
    clipShapefile(input_shp, output_shp, clip_shp)

    current_step += 1

    show_progress(current_step, total_steps, "Rasterizing beaches subset...")

    #logging.info('Rasterizing beaches subset...')
    #input_shp = str(pathlib.Path(os.path.join(temp_dir, 'clip_bb300_r.shp')))
    #output_raster = str(pathlib.Path(os.path.join(temp_dir, 'clip_bb300_r.tif')))
    #logging.info('Rasterizing beaches subset...')
    input_shp = os.path.join(temp_dir, f"{date_folder}_clip_bb300_r.shp")
    output_raster = os.path.join(temp_dir, f"{date_folder}_clip_bb300_r.tif")
    #raster_template = getBandData20mFromScene(scene_path, 'B11')
    raster_template = band_paths['B11']

    codes = [8192,8193,8194,8195,8196,8197,8198,8199,9200,8201,8202,8203,8204,8205,8206,8207,8208,8209,8210,8211,8212,8213,8214,8215]
    bc = "(" + ",".join(str(c) for c in codes) + ")"
    where_clause = f"BEACH_CODE IN ({bc})"

    #rasterizeShapefile(input_shp, output_raster, raster_template, bc)
    if os.path.exists(input_shp):
        rasterizeShapefile(input_shp, output_raster, raster_template, "BEACH_CODE", where=where_clause)
    else:
        print(f"ERREUR : le fichier {input_shp} n'existe pas")


    current_step += 1

    show_progress(current_step, total_steps, "Masking rough pixel line with beaches subset...")

    #logging.info('Masking rough pixel line with beaches subset...')
    # Trait de côte pixelisé
    pixel_line_tif = os.path.join(temp_dir, f"{date_folder}_pixel_line.tif")

    # Raster masque des plages (issu de la rasterisation du shapefile clippé

    beach_mask_tif = str(pathlib.Path(os.path.join(temp_dir, 'clip_bb300_r.tif')))

    # Vérifier l alignement des deux entrer pixel_line et beach_mask_tif

    #check_raster_alignment(pixel_line_tif, beach_mask_tif)

    # Alignement du raster plage sur le raster pixel_line

    beach_mask_aligned_tif = os.path.join(temp_dir, f"{date_folder}_clip_bb300_r_aligned.tif")

    if not os.path.exists(beach_mask_aligned_tif):
        print(f"ERREUR : Le raster aligné {beach_mask_aligned_tif} n'a pas été créé !")
        # Ici, tu peux relancer align_raster_to_template ou afficher un message d'erreur détaillé
    else:
        print("Le raster aligné existe bien.")


    align_raster_to_template(beach_mask_tif, pixel_line_tif, beach_mask_aligned_tif)

    # Utilisation du raster aligné pour le masquage

    maskPixelLine(pixel_line_tif, beach_mask_aligned_tif)

    current_step += 1

    show_progress(current_step, total_steps, "Extracting points...")

    #logging.info('Extracting points...')
    swir1_path = band_paths['B11']
    pixel_line_masked = os.path.join(temp_dir, f"{date_folder}_pixel_line.tif")
    processing_path = temp_dir
    kernel_size = 3
    ppp = 4
    degree = 3
    res = extractPoints(
        swir1_path,
        pixel_line_masked,
        processing_path,
        int(kernel_size),
        int(ppp),
        int(degree)
        )

    current_step += 1

    if res:
        show_progress(current_step, total_steps, "Computing average points...")
        #logging.info('Computing average points...')
        averagePoints(swir1_path, processing_path, 50, 3)

        current_step += 1

        show_progress(current_step, total_steps, "Making point shp...")
        #logging.info('Making point shp...')

        shp_path_average = createShpFromAverageFile(swir1_path, processing_path)

        current_step += 1

        show_progress(current_step, total_steps, "Transfering beaches identifiers...")
        #logging.info('Transfering beaches identifiers...')

        clip_beach_shp = str(pathlib.Path(os.path.join(processing_path, 'clip_bb300_r.shp')))

        copyShpIdentifiers(clip_beach_shp, shp_path_average)

        current_step += 1

        show_progress(current_step, total_steps, "Cleaning points and making final shoreline in line vector format...")
        #logging.info('Cleaning points and making final shoreline in line vector format...')

        cleanPoints2(shp_path_average, 150, 1)

        current_step += 1

        show_progress(current_step, total_steps, "Export final shoreline shapefiles to SDS folder...")
        #logging.info('Export final shoreline shapefiles to SDS folder...')

        sds_path = os.path.join(scene_path, 'SDS')

        if not os.path.exists(sds_path):
            os.makedirs(sds_path)
            copyShpToFolder(processing_path, sds_path, target_epsg)

        else:
            logging.warning('No reults in extraction points process.')
            sys.exit(1)

        current_step += 1

    print(f"Indices et shapefiles exportés pour {date_folder}")

print("Traitement terminé !")
