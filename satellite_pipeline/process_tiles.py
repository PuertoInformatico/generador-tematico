"""
process_tiles.py  – Extrae bandas de cada ZIP y las recorta a Madre de Dios,
agrupando los resultados por código de tile Sentinel-2.

Entrada:  raw_scenes/<product>.zip
Salida:   processed_tiles/tile_<CODE>/<YYYYMMDD>_<BANDA>.tif

Ejemplo de nombre de producto:
  S2A_MSIL2A_20240301T150731_N0510_R082_T19LDF_20240301T213059
  → tile 19LDF, fecha 20240301
  → processed_tiles/tile_19LDF/20240301_B02.tif
                               20240301_B03.tif  ...
"""

import os
import re
import sys
import tempfile
import zipfile
from pathlib import Path

for _var in ("PROJ_LIB", "PROJ_DATA", "GDAL_DATA"):
    os.environ.pop(_var, None)

import fiona
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.mask import mask as rio_mask
from rasterio.transform import from_bounds
from shapely.geometry import shape, mapping
from shapely.ops import transform as shp_transform
from pyproj import Transformer

ROOT       = Path(__file__).parent.parent
INPUT_DIR  = ROOT / "raw_scenes"
OUTPUT_DIR = ROOT / "processed_tiles"
SHAPEFILE  = Path(__file__).parent / "mdd" / "mdd.shp"

TARGET_BANDS   = ["B02", "B03", "B04", "B08"]
OPTIONAL_BANDS = ["SCL"]


# ── Utilidades ────────────────────────────────────────────────────────────────

def extract_tile_code(name: str) -> str | None:
    """Extrae el código de tile (ej. '19LDF') del nombre del producto."""
    m = re.search(r'_T(\d{2}[A-Z]{3})_', name)
    return m.group(1) if m else None


def extract_date(name: str) -> str | None:
    """Extrae la fecha de adquisición YYYYMMDD del nombre del producto."""
    m = re.search(r'MSI\w+_(\d{8})T', name)
    return m.group(1) if m else None


def infer_crs(tile_code: str) -> CRS:
    """Devuelve el CRS UTM Sur para el código de tile dado (Madre de Dios = hemisferio Sur)."""
    zone = int(tile_code[:2])
    return CRS.from_epsg(32700 + zone)


def load_shapes(shp_path: Path, target_crs: CRS) -> list:
    """Lee el shapefile y reproyecta las geometrías al CRS objetivo."""
    with fiona.open(shp_path) as src:
        src_crs   = CRS.from_user_input(src.crs)
        raw_geoms = [feat["geometry"] for feat in src]
    if src_crs == target_crs:
        return raw_geoms
    tr = Transformer.from_crs(src_crs, target_crs, always_xy=True)
    return [mapping(shp_transform(tr.transform, shape(g))) for g in raw_geoms]


def find_jp2(zf: zipfile.ZipFile, bands: list, resolution: str) -> dict:
    """Devuelve {banda: entry_path} para las bandas pedidas en la resolucion dada."""
    found = {}
    for entry in zf.namelist():
        p = Path(entry)
        if "IMG_DATA" in p.parts and resolution in p.parts and p.suffix == ".jp2":
            for band in bands:
                if f"_{band}_" in p.name or p.stem.endswith(f"_{band}"):
                    found[band] = entry
    return found


# ── Procesamiento ─────────────────────────────────────────────────────────────

def clip_band(jp2_path: Path, out_path: Path, tile_crs: CRS,
              geoms: list, ref_shape: tuple | None) -> bool:
    """
    Recorta un JP2 con las geometrias de MDD y guarda como GeoTIFF.
    Si ref_shape se especifica (para SCL), remuestrea primero al tamaño de referencia.
    """
    try:
        with rasterio.open(jp2_path) as src:
            file_crs = src.crs or tile_crs

            if ref_shape and src.shape != ref_shape:
                # Remuestrear SCL (20m → 10m) antes de recortar
                data          = src.read(1, out_shape=ref_shape, resampling=Resampling.nearest)
                new_transform = from_bounds(*src.bounds, ref_shape[1], ref_shape[0])
                profile       = src.profile.copy()
                profile.update(crs=file_crs, height=ref_shape[0],
                               width=ref_shape[1], transform=new_transform)
                tmp_rs = jp2_path.with_suffix(".tmp.tif")
                with rasterio.open(tmp_rs, "w", **profile) as t:
                    t.write(data, 1)
                read_path = tmp_rs
            else:
                read_path = jp2_path

        with rasterio.open(read_path) as src2:
            file_crs = src2.crs or tile_crs
            out_img, out_transform = rio_mask(src2, geoms, crop=True)
            profile = src2.profile.copy()

        profile.update(
            driver="GTiff", crs=file_crs,
            height=out_img.shape[1], width=out_img.shape[2],
            transform=out_transform,
            compress="lzw", tiled=True, blockxsize=256, blockysize=256,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(out_img)

        if read_path != jp2_path and read_path.exists():
            read_path.unlink()
        return True

    except Exception as e:
        print(f"    [ERROR] {out_path.name}: {e}")
        return False


def process_zip(zip_path: Path):
    name      = zip_path.stem
    tile_code = extract_tile_code(name)
    date      = extract_date(name)

    if not tile_code or not date:
        print(f"  [AVISO] No se pudo extraer tile/fecha de: {name}")
        return

    tile_dir = OUTPUT_DIR / f"tile_{tile_code}"
    tile_dir.mkdir(parents=True, exist_ok=True)
    tile_crs = infer_crs(tile_code)

    print(f"\n{name}")
    print(f"  tile={tile_code}  fecha={date}  CRS=EPSG:{tile_crs.to_epsg()}")

    # Omitir si todas las bandas ya existen
    expected = [tile_dir / f"{date}_{b}.tif" for b in TARGET_BANDS]
    if all(p.exists() for p in expected):
        print(f"  [omitido] ya procesado.")
        return

    geoms = load_shapes(SHAPEFILE, tile_crs)

    with zipfile.ZipFile(zip_path) as zf:
        bands_10m = find_jp2(zf, TARGET_BANDS,   "R10m")
        bands_20m = find_jp2(zf, OPTIONAL_BANDS, "R20m")
        all_bands = {**bands_10m, **bands_20m}

        if not all_bands:
            print("  [AVISO] No se encontraron bandas en el ZIP.")
            return

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # Extraer todos los JP2 al directorio temporal
            for band, entry in all_bands.items():
                jp2 = tmp / Path(entry).name
                with zf.open(entry) as src:
                    jp2.write_bytes(src.read())

            # ref_shape = dimensiones de B04 (10m), necesario para remuestrear SCL
            ref_shape = None
            if "B04" in bands_10m:
                b04_jp2 = tmp / Path(bands_10m["B04"]).name
                with rasterio.open(b04_jp2) as src:
                    ref_shape = src.shape

            for band, entry in all_bands.items():
                jp2 = tmp / Path(entry).name
                out = tile_dir / f"{date}_{band}.tif"
                if out.exists():
                    print(f"  [omitido] {out.name}")
                    continue
                print(f"  {band} → {out.name}")
                clip_band(jp2, out, tile_crs, geoms,
                          ref_shape if band in OPTIONAL_BANDS else None)

    print(f"  Guardado en: {tile_dir}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not SHAPEFILE.exists():
        print(f"Shapefile no encontrado: {SHAPEFILE}")
        sys.exit(1)
    if not INPUT_DIR.exists():
        print(f"Carpeta raw_scenes no encontrada: {INPUT_DIR}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    zips = sorted(INPUT_DIR.glob("*.zip"))
    if not zips:
        print(f"No se encontraron .zip en {INPUT_DIR}")
        sys.exit(0)

    print(f"Encontrados {len(zips)} ZIP(s).\n")
    for zp in zips:
        process_zip(zp)

    print("\nProceso completado.")


if __name__ == "__main__":
    main()
