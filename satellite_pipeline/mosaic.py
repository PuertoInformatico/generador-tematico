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

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_origin
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

def warp_tile(src_path: Path, dst_crs, dst_transform,
              dst_w: int, dst_h: int) -> np.ndarray:
    """Reproyecta las 4 bandas del tile a la grilla destino. Retorna [4, H, W]."""
    n = len(BAND_ORDER)
    dst = np.zeros((n, dst_h, dst_w), dtype="float32")
    with rasterio.open(src_path) as src:
        for b in range(1, n + 1):
            reproject(
                source=rasterio.band(src, b),
                destination=dst[b - 1],
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
                dst_nodata=0.0,
            )
    return dst


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not INPUT_DIR.exists():
        print(f"Carpeta composite_tiles no encontrada: {INPUT_DIR}")
        sys.exit(1)

    tiles = sorted(INPUT_DIR.glob("*.tif"))
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

    # Mosaico con estrategia "primer pixel válido": rellena huecos progresivamente
    mosaic = np.zeros((len(BAND_ORDER), dst_h, dst_w), dtype="float32")

    for tile_path in tiles:
        print(f"Aplicando: {tile_path.name}")
        warped = warp_tile(tile_path, dst_crs, dst_transform, dst_w, dst_h)
        # Solo rellenar donde el mosaico aún no tiene datos
        empty = mosaic[0] == 0
        for b in range(len(BAND_ORDER)):
            mosaic[b][empty] = warped[b][empty]

    pct = 100.0 * (mosaic[0] > 0).sum() / (dst_h * dst_w)
    print(f"\nCobertura del mosaico: {pct:.1f}%")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff", "dtype": "float32",
        "crs": dst_crs, "transform": dst_transform,
        "width": dst_w, "height": dst_h,
        "count": len(BAND_ORDER),
        "compress": "lzw", "tiled": True,
        "blockxsize": 256, "blockysize": 256, "nodata": 0,
    }
    with rasterio.open(OUTPUT_PATH, "w", **profile) as dst:
        dst.write(mosaic)
        for i, (name, band) in enumerate(zip(BAND_NAMES, BAND_ORDER), start=1):
            dst.update_tags(i, name=name, band=band)

    print(f"Mosaico guardado: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
