"""
composite_tiles.py  – Genera un composite libre de nubes por tile Sentinel-2.

Entrada:  processed_tiles/tile_<CODE>/<YYYYMMDD>_<BANDA>.tif
Salida:   composite_tiles/tile_<CODE>.tif  (4 bandas: B02, B03, B04, B08)

Proceso por tile:
  1. Agrupa archivos por fecha (prefijo YYYYMMDD).
  2. Para cada fecha carga las 4 bandas + SCL.
  3. Marca como NaN los pixeles con nube/sombra/sin dato (SCL).
  4. Apila todas las fechas y aplica mediana por pixel (ignora NaN).

Valores SCL descartados:
  0 No Data | 1 Saturado | 3 Sombra | 8/9 Nube | 10 Cirrus
"""

import os
import re
import sys
from pathlib import Path

for _var in ("PROJ_LIB", "PROJ_DATA", "GDAL_DATA"):
    os.environ.pop(_var, None)

import numpy as np
import rasterio
from rasterio.enums import Resampling

ROOT       = Path(__file__).parent.parent
INPUT_DIR  = ROOT / "processed_tiles"
OUTPUT_DIR = ROOT / "composite_tiles"

BAND_ORDER  = ["B02", "B03", "B04", "B08"]
BAND_NAMES  = ["Azul", "Verde", "Rojo", "NIR"]
SCL_INVALID = {0, 1, 3, 8, 9, 10}


# ── Utilidades ────────────────────────────────────────────────────────────────

def get_dates(tile_dir: Path) -> list[str]:
    """Devuelve fechas únicas (YYYYMMDD) disponibles en el directorio del tile."""
    dates = set()
    for f in tile_dir.glob("????????_*.tif"):
        m = re.match(r'^(\d{8})_', f.name)
        if m:
            dates.add(m.group(1))
    return sorted(dates)



def load_mask(tile_dir: Path, date: str, ref_shape: tuple) -> np.ndarray:
    """Carga la máscara SCL para una fecha. Retorna array bool [H, W]."""
    scl_path = tile_dir / f"{date}_SCL.tif"
    if not scl_path.exists():
        print(f"    [AVISO] SCL no encontrada para {date}. Sin máscara de nubes.")
        return np.ones(ref_shape, dtype=bool)
    with rasterio.open(scl_path) as src:
        scl_raw = (src.read(1, out_shape=ref_shape, resampling=Resampling.nearest)
                   if src.shape != ref_shape else src.read(1))
    return ~np.isin(scl_raw.astype("uint8"), list(SCL_INVALID))


def get_valid_dates(tile_dir: Path) -> tuple[list[str], tuple, dict]:
    """Filtra fechas con todas las bandas disponibles. Retorna (fechas, ref_shape, ref_profile)."""
    valid, ref_shape, ref_profile = [], None, None
    for date in get_dates(tile_dir):
        paths = [tile_dir / f"{date}_{b}.tif" for b in BAND_ORDER]
        if not all(p.exists() for p in paths):
            print(f"    [AVISO] Faltan bandas para {date}. Omitida.")
            continue
        if ref_shape is None:
            with rasterio.open(paths[0]) as src:
                ref_shape   = src.shape
                ref_profile = src.profile.copy()
        valid.append(date)
    return valid, ref_shape, ref_profile


# ── Composite por tile (banda a banda para minimizar RAM) ────────────────────

def is_stale(out_path: Path, input_dir: Path) -> bool:
    """True si el output no existe o hay algún input más nuevo."""
    if not out_path.exists():
        return True
    out_mtime    = out_path.stat().st_mtime
    input_mtimes = [f.stat().st_mtime for f in input_dir.glob("????????_*.tif")]
    return bool(input_mtimes) and max(input_mtimes) > out_mtime


def composite_tile(tile_dir: Path) -> bool:
    tile_name = tile_dir.name                    # "tile_19LDF"
    out_path  = OUTPUT_DIR / f"{tile_name}.tif"

    if not is_stale(out_path, tile_dir):
        print(f"  [omitido] actualizado: {out_path.name}")
        return True

    all_dates = get_dates(tile_dir)
    if not all_dates:
        print(f"  [AVISO] No se encontraron fechas en {tile_dir}")
        return False

    valid_dates, ref_shape, ref_profile = get_valid_dates(tile_dir)
    if not valid_dates:
        print(f"  No se pudo cargar ninguna fecha para {tile_name}.")
        return False

    print(f"\n{tile_name}: {len(valid_dates)} fecha(s) válidas de {len(all_dates)}")

    # Cargar todas las máscaras SCL de una vez (son ligeras: [N, H, W] bool)
    masks = []
    for date in valid_dates:
        m = load_mask(tile_dir, date, ref_shape)
        pct = 100.0 * m.sum() / m.size
        print(f"    {date}: {pct:.1f}% de pixeles válidos")
        masks.append(m)

    H, W = ref_shape
    N    = len(valid_dates)
    composite = np.zeros((len(BAND_ORDER), H, W), dtype="float32")

    # Procesar banda a banda → pico de RAM = N×H×W×4 bytes (sin el factor ×4 bandas)
    for bi, band in enumerate(BAND_ORDER):
        band_cube = np.full((N, H, W), np.nan, dtype="float32")
        for di, (date, mask) in enumerate(zip(valid_dates, masks)):
            with rasterio.open(tile_dir / f"{date}_{band}.tif") as src:
                arr = src.read(1).astype("float32")
            arr[~mask] = np.nan
            band_cube[di] = arr
        composite[bi] = np.nan_to_num(np.nanmedian(band_cube, axis=0), nan=0.0)
        del band_cube   # liberar RAM antes de la siguiente banda

    pct = 100.0 * (composite[0] > 0).sum() / composite[0].size
    print(f"  Cobertura final: {pct:.1f}%")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ref_profile.update(
        count=len(BAND_ORDER), dtype="float32",
        compress="lzw", tiled=True, blockxsize=256, blockysize=256, nodata=0,
    )
    with rasterio.open(out_path, "w", **ref_profile) as dst:
        dst.write(composite)
        for i, (name, band) in enumerate(zip(BAND_NAMES, BAND_ORDER), start=1):
            dst.update_tags(i, name=name, band=band)

    print(f"  Guardado: {out_path}")
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not INPUT_DIR.exists():
        print(f"Carpeta processed_tiles no encontrada: {INPUT_DIR}")
        sys.exit(1)

    tiles = sorted(p for p in INPUT_DIR.iterdir() if p.is_dir())
    if not tiles:
        print(f"No se encontraron tiles en {INPUT_DIR}")
        sys.exit(0)

    print(f"Generando composite para {len(tiles)} tile(s)...\n")
    for tile_dir in tiles:
        composite_tile(tile_dir)

    print("\nProceso completado.")


if __name__ == "__main__":
    main()
