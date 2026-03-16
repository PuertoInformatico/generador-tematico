"""
mosaic.py  – Une los composites de todos los tiles en un mosaico regional.

Entrada:  composite_tiles/tile_<CODE>.tif   (cada uno en su UTM nativo)
Salida:   regional_mosaic/madre_dios.tif    (4 bandas, EPSG:32719, 10 m)

Proceso:
  1. Calcula el bounding box unión de todos los tiles en EPSG:32719.
  2. Reproyecta cada tile a esa grilla común.
  3. Combina con estrategia "primer pixel válido" (rellena huecos entre tiles).
"""

import os
import sys
from pathlib import Path

for _var in ("PROJ_LIB", "PROJ_DATA", "GDAL_DATA"):
    os.environ.pop(_var, None)

import math

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.windows import Window
from rasterio.warp import reproject, transform_bounds

ROOT       = Path(__file__).parent.parent
INPUT_DIR  = ROOT / "composite_tiles"
OUTPUT_DIR = ROOT / "regional_mosaic"
OUTPUT_PATH = OUTPUT_DIR / "madre_dios.tif"

TARGET_EPSG = 32719
RESOLUTION  = 10.0
BAND_ORDER  = ["B02", "B03", "B04", "B08"]
BAND_NAMES  = ["Azul", "Verde", "Rojo", "NIR"]


# ── Grilla común ──────────────────────────────────────────────────────────────

def get_common_grid(tile_paths: list[Path]) -> tuple:
    """Calcula bounding box unión de todos los tiles en TARGET_EPSG."""
    dst_crs = CRS.from_epsg(TARGET_EPSG)
    lefts, bottoms, rights, tops = [], [], [], []
    for p in tile_paths:
        with rasterio.open(p) as src:
            b = transform_bounds(src.crs, dst_crs, *src.bounds)
        lefts.append(b[0]);  bottoms.append(b[1])
        rights.append(b[2]); tops.append(b[3])

    left, bottom = min(lefts), min(bottoms)
    right, top   = max(rights), max(tops)
    transform    = from_origin(left, top, RESOLUTION, RESOLUTION)
    width        = int(round((right - left) / RESOLUTION))
    height       = int(round((top - bottom) / RESOLUTION))
    return dst_crs, transform, width, height


# ── Reproyección de un tile ───────────────────────────────────────────────────

def warp_tile_to_window(
    src_path: Path,
    dst_crs,
    dst_transform,
    dst_w: int,
    dst_h: int,
) -> tuple:
    """
    Reproyecta un tile solo a su ventana dentro de la grilla destino.
    Retorna (array [4, h, w], Window) sin cargar la grilla completa en RAM.
    Retorna (None, None) si el tile no solapa con la grilla.
    """
    n = len(BAND_ORDER)

    with rasterio.open(src_path) as src:
        bounds_dst = transform_bounds(src.crs, dst_crs, *src.bounds)
    left, bottom, right, top = bounds_dst

    col_off = max(0,    int(math.floor((left            - dst_transform.c) / RESOLUTION)))
    row_off = max(0,    int(math.floor((dst_transform.f - top)             / RESOLUTION)))
    col_end = min(dst_w, int(math.ceil( (right           - dst_transform.c) / RESOLUTION)))
    row_end = min(dst_h, int(math.ceil( (dst_transform.f - bottom)          / RESOLUTION)))

    win_w = col_end - col_off
    win_h = row_end - row_off
    if win_w <= 0 or win_h <= 0:
        return None, None

    win_left      = dst_transform.c + col_off * RESOLUTION
    win_top       = dst_transform.f - row_off * RESOLUTION
    win_transform = from_origin(win_left, win_top, RESOLUTION, RESOLUTION)

    dst_arr = np.zeros((n, win_h, win_w), dtype="float32")
    with rasterio.open(src_path) as src:
        for b in range(1, n + 1):
            reproject(
                source=rasterio.band(src, b),
                destination=dst_arr[b - 1],
                dst_transform=win_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
                dst_nodata=0.0,
            )
    return dst_arr, Window(col_off, row_off, win_w, win_h)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not INPUT_DIR.exists():
        print(f"Carpeta composite_tiles no encontrada: {INPUT_DIR}")
        sys.exit(1)

    # Ordenar de más antiguo a más reciente: el dato más nuevo sobreescribe al final
    tiles = sorted(INPUT_DIR.glob("*.tif"), key=lambda p: p.stat().st_mtime)
    if not tiles:
        print(f"No se encontraron tiles en {INPUT_DIR}")
        sys.exit(0)

    if OUTPUT_PATH.exists():
        out_mtime    = OUTPUT_PATH.stat().st_mtime
        newest_tile  = max(p.stat().st_mtime for p in tiles)
        if out_mtime >= newest_tile:
            print(f"[omitido] mosaico actualizado: {OUTPUT_PATH.name}")
            return

    print(f"Mosaicando {len(tiles)} tile(s) → EPSG:{TARGET_EPSG}, {RESOLUTION:.0f} m\n")
    for t in tiles:
        print(f"  {t.name}")

    dst_crs, dst_transform, dst_w, dst_h = get_common_grid(tiles)
    print(f"\nGrilla final: {dst_w} x {dst_h} px  "
          f"({dst_w * RESOLUTION / 1000:.0f} x {dst_h * RESOLUTION / 1000:.0f} km)\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff", "dtype": "float32",
        "crs": dst_crs, "transform": dst_transform,
        "width": dst_w, "height": dst_h,
        "count": len(BAND_ORDER),
        "compress": "lzw", "tiled": True,
        "blockxsize": 256, "blockysize": 256, "nodata": 0,
        "bigtiff": "YES",
    }

    # Crear el archivo de salida vacío en disco (sin alocar la grilla completa en RAM)
    with rasterio.open(OUTPUT_PATH, "w", **profile) as dst_file:
        for i, (name, band) in enumerate(zip(BAND_NAMES, BAND_ORDER), start=1):
            dst_file.update_tags(i, name=name, band=band)

    # Aplicar tiles: rellena vacíos Y actualiza con dato válido más reciente
    # Como tiles está ordenado de antiguo→reciente, cada tile nuevo sobreescribe
    # donde tiene dato válido (!= 0), reflejando el estado actual del terreno.
    with rasterio.open(OUTPUT_PATH, "r+") as dst_file:
        for tile_path in tiles:
            print(f"Aplicando: {tile_path.name}")
            warped, window = warp_tile_to_window(
                tile_path, dst_crs, dst_transform, dst_w, dst_h
            )
            if warped is None:
                continue
            existing = dst_file.read(window=window)
            valid_new = warped[0] != 0   # píxeles válidos del tile nuevo
            for b in range(len(BAND_ORDER)):
                existing[b][valid_new] = warped[b][valid_new]
            dst_file.write(existing, window=window)

    # Calcular cobertura leyendo bloque a bloque (evita cargar la imagen completa)
    valid = 0
    with rasterio.open(OUTPUT_PATH) as dst_file:
        for _, win in dst_file.block_windows(1):
            valid += int((dst_file.read(1, window=win) > 0).sum())
    pct = 100.0 * valid / (dst_h * dst_w)

    print(f"\nCobertura del mosaico: {pct:.1f}%")
    print(f"Mosaico guardado: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
