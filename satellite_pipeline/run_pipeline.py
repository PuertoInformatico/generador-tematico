"""
run_pipeline.py  – Ejecuta el pipeline completo de forma incremental.

Uso:
    python satellite_pipeline/run_pipeline.py

Cada paso solo reejecutará el trabajo si sus entradas son más nuevas que
sus salidas, por lo que es seguro ejecutarlo con cualquier frecuencia.

Flujo:
    raw_scenes/          ← download.py
    processed_tiles/     ← process_tiles.py
    composite_tiles/     ← composite_tiles.py
    regional_mosaic/     ← mosaic.py + classify.py
"""

import sys
import importlib
from pathlib import Path

# Añadir el directorio del pipeline al path para poder importar los módulos
sys.path.insert(0, str(Path(__file__).parent))

STEPS = [
    ("download",         "Descargando escenas nuevas"),
    ("process_tiles",    "Procesando tiles (extracción + recorte)"),
    ("composite_tiles",  "Generando composites por tile"),
    ("mosaic",           "Construyendo mosaico regional"),
    ("classify",         "Clasificando cobertura (NDVI)"),
]


def run():
    print("=" * 60)
    print("  PIPELINE SENTINEL-2  —  Madre de Dios")
    print("=" * 60)

    for module_name, label in STEPS:
        print(f"\n{'─' * 60}")
        print(f"  {label}")
        print(f"{'─' * 60}")
        try:
            mod = importlib.import_module(module_name)
            # Forzar recarga por si se ejecuta el pipeline varias veces
            importlib.reload(mod)
            mod.main()
        except SystemExit as e:
            if e.code != 0:
                print(f"\n[ERROR] El paso '{module_name}' terminó con código {e.code}.")
                sys.exit(e.code)
        except Exception as e:
            print(f"\n[ERROR] Fallo en '{module_name}': {e}")
            raise

    print(f"\n{'=' * 60}")
    print("  Pipeline completado.")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    run()
