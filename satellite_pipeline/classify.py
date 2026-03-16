"""
classify.py  – Calcula NDVI y clasifica la cobertura sobre el mosaico regional.

Entrada:  regional_mosaic/madre_dios.tif   (4 bandas: B02, B03, B04, B08)
Salida:   regional_mosaic/landcover.tif    (uint8)

Categorías NDVI:
  1 = Bosque          (NDVI > 0.6)
  2 = Vegetación      (0.3 < NDVI ≤ 0.6)
  3 = Suelo           (0.1 < NDVI ≤ 0.3)
  4 = Agua / Minería  (NDVI ≤ 0.1)
  0 = Sin dato
"""

import os
import sys
from pathlib import Path

for _var in ("PROJ_LIB", "PROJ_DATA", "GDAL_DATA"):
    os.environ.pop(_var, None)

import numpy as np
import rasterio

ROOT        = Path(__file__).parent.parent
INPUT_PATH  = ROOT / "regional_mosaic" / "madre_dios.tif"
OUTPUT_PATH = ROOT / "regional_mosaic" / "landcover.tif"

# Índices de banda dentro del stack (1-based): B02=1, B03=2, B04=3, B08=4
BAND_RED = 3   # B04
BAND_NIR = 4   # B08

THRESHOLDS = {"bosque": 0.6, "vegetacion": 0.3, "suelo": 0.1}

LABELS = {0: "Sin dato", 1: "Bosque", 2: "Vegetación",
          3: "Suelo", 4: "Agua / Minería"}


def main():
    if not INPUT_PATH.exists():
        print(f"Archivo no encontrado: {INPUT_PATH}")
        sys.exit(1)

    if OUTPUT_PATH.exists():
        if OUTPUT_PATH.stat().st_mtime >= INPUT_PATH.stat().st_mtime:
            print(f"[omitido] clasificación actualizada: {OUTPUT_PATH.name}")
            return

    print(f"Clasificando: {INPUT_PATH.name}")

    with rasterio.open(INPUT_PATH) as src:
        profile = src.profile.copy()
        nodata  = src.nodata or 0

    profile.update(
        count=1, dtype="uint8", nodata=0,
        compress="lzw", tiled=True, blockxsize=256, blockysize=256,
    )

    counts = {cat: 0 for cat in LABELS}
    total  = 0

    with rasterio.open(INPUT_PATH) as src, rasterio.open(OUTPUT_PATH, "w", **profile) as dst:
        dst.update_tags(1, categories="0=sin_dato,1=bosque,2=vegetacion,3=suelo,4=agua_mineria")
        windows = list(src.block_windows(1))
        n_wins  = len(windows)
        for idx, (_, win) in enumerate(windows, 1):
            if idx % 500 == 0 or idx == n_wins:
                print(f"  bloque {idx}/{n_wins}", end="\r")

            red = src.read(BAND_RED, window=win).astype("float32")
            nir = src.read(BAND_NIR, window=win).astype("float32")

            no_data_mask = (red == nodata) & (nir == nodata)

            denom = nir + red
            with np.errstate(invalid="ignore", divide="ignore"):
                ndvi = np.where(denom == 0, np.nan, (nir - red) / denom)

            lc = np.zeros(red.shape, dtype="uint8")
            lc[ndvi > THRESHOLDS["bosque"]]                                             = 1
            lc[(ndvi > THRESHOLDS["vegetacion"]) & (ndvi <= THRESHOLDS["bosque"])]     = 2
            lc[(ndvi > THRESHOLDS["suelo"])      & (ndvi <= THRESHOLDS["vegetacion"])] = 3
            lc[(~np.isnan(ndvi)) & (ndvi <= THRESHOLDS["suelo"])]                      = 4
            lc[no_data_mask] = 0

            dst.write(lc[np.newaxis, ...], window=win)

            for cat in LABELS:
                counts[cat] += int((lc == cat).sum())
            total += lc.size

    print()  # salto de línea tras el contador
    for cat, label in LABELS.items():
        n = counts[cat]
        print(f"  {label:16s} {100 * n / total:5.1f}%  ({n:,} px)")

    print(f"\nLandcover guardado: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
