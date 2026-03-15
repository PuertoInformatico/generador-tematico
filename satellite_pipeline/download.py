"""
download.py  – Descarga escenas Sentinel-2 L2A desde Copernicus Data Space.
Salida: raw_scenes/<product_id>.zip
"""

import os
import re
import sys
import requests
from pathlib import Path
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

load_dotenv()

ROOT       = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "raw_scenes"

BBOX       = [-72.4, -13.2, -68.6, -10.0]   # Madre de Dios
MAX_CLOUD  = 60        # % cobertura de nubes — la SCL filtra pixel a pixel (60 ideal para amazonía)
DAYS_BACK  = 90        # 90 días ≈ ~18 pasadas por tile (revisita 5 días)
MAX_SCENES = 60        # suficiente para cubrir todos los tiles del área


# ── Autenticación ─────────────────────────────────────────────────────────────

def get_token() -> str:
    url  = ("https://identity.dataspace.copernicus.eu/auth/realms/"
            "CDSE/protocol/openid-connect/token")
    data = {
        "client_id":  "cdse-public",
        "username":   os.getenv("COPERNICUS_USERNAME"),
        "password":   os.getenv("COPERNICUS_PASSWORD"),
        "grant_type": "password",
    }
    r = requests.post(url, data=data)
    if r.status_code != 200:
        raise Exception(f"Error al obtener token: {r.status_code} {r.text}")
    token = r.json().get("access_token")
    if not token:
        raise KeyError("No se encontró 'access_token' en la respuesta.")
    print("Token obtenido.")
    return token


# ── Búsqueda ──────────────────────────────────────────────────────────────────

def search_scenes(token: str) -> list[dict]:
    now   = datetime.now(timezone.utc)
    start = now - timedelta(days=DAYS_BACK)
    query = {
        "collections": ["sentinel-2-l2a"],
        "limit":   MAX_SCENES,
        "bbox":    BBOX,
        "datetime": (f"{start.strftime('%Y-%m-%dT%H:%M:%SZ')}/"
                     f"{now.strftime('%Y-%m-%dT%H:%M:%SZ')}"),
        "query":   {"eo:cloud_cover": {"lt": MAX_CLOUD}},
    }
    r = requests.post(
        "https://catalogue.dataspace.copernicus.eu/stac/search",
        headers={"Authorization": f"Bearer {token}",
                 "Content-Type": "application/json"},
        json=query,
    )
    if r.status_code != 200:
        raise Exception(f"Error en búsqueda: {r.status_code} {r.text}")
    features = r.json().get("features", [])
    print(f"Escenas encontradas: {len(features)}")
    for f in features:
        print(f"  {f['id']}")
    return features


# ── Descarga ──────────────────────────────────────────────────────────────────

def download_scene(product_id: str, url: str, token: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"{product_id}.zip"
    if out.exists():
        print(f"  [omitido] ya existe: {out.name}")
        return
    print(f"  Descargando {out.name} ...")
    r = requests.get(url, headers={"Authorization": f"Bearer {token}"}, stream=True)
    if r.status_code != 200:
        print(f"  [ERROR] {r.status_code}: {r.text}")
        return
    with open(out, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"  Guardado: {out}")


# ── Limpieza de escenas fuera de la ventana temporal ─────────────────────────

def scene_date(zip_path: Path) -> datetime | None:
    """Extrae la fecha de adquisición del nombre del producto."""
    m = re.search(r'MSI\w+_(\d{8})T', zip_path.stem)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y%m%d").replace(tzinfo=timezone.utc)


def cleanup_old_scenes():
    """
    Elimina ZIPs y tiles procesados cuya fecha de adquisición queda fuera
    de la ventana DAYS_BACK. Mantiene el disco acotado en ejecuciones periódicas.
    """
    if not OUTPUT_DIR.exists():
        return

    cutoff      = datetime.now(timezone.utc) - timedelta(days=DAYS_BACK)
    tiles_root  = ROOT / "processed_tiles"
    removed_zip = 0
    removed_tif = 0

    for zip_path in sorted(OUTPUT_DIR.glob("*.zip")):
        acq = scene_date(zip_path)
        if acq is None or acq >= cutoff:
            continue

        # Eliminar ZIP
        zip_path.unlink()
        removed_zip += 1
        print(f"  [limpieza] ZIP eliminado: {zip_path.name}")

        # Eliminar TIFs procesados correspondientes (YYYYMMDD_*.tif en cualquier tile)
        date_str   = acq.strftime("%Y%m%d")
        tile_code  = None
        m = re.search(r'_T(\d{2}[A-Z]{3})_', zip_path.stem)
        if m:
            tile_code = m.group(1)
            tile_dir  = tiles_root / f"tile_{tile_code}"
            for tif in tile_dir.glob(f"{date_str}_*.tif"):
                tif.unlink()
                removed_tif += 1
            print(f"  [limpieza] TIFs de {date_str} en tile_{tile_code} eliminados")

    if removed_zip:
        print(f"\n  Limpieza: {removed_zip} ZIP(s) y {removed_tif} TIF(s) fuera de la ventana de {DAYS_BACK} días.")
    else:
        print(f"  Sin escenas fuera de la ventana de {DAYS_BACK} días.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    token    = get_token()
    features = search_scenes(token)
    for item in features:
        asset = item["assets"].get("Product")
        if not asset:
            print(f"  Sin asset 'Product' en {item['id']}. Saltando.")
            continue
        download_scene(item["id"], asset["href"], token)

    print("\nLimpiando escenas fuera de la ventana temporal...")
    cleanup_old_scenes()
    print("\nDescarga completada.")


if __name__ == "__main__":
    main()
