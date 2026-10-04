"""Run the original frozen Stage-2 v2 scientific script unchanged.
Only hardens optional package-version logging: eccodes is needed for raw GRIB preparation,
not for replaying the already-prepared NPZ benchmark bundled here.
"""
from pathlib import Path
import importlib.metadata, runpy, os
HERE=Path(__file__).resolve().parent
_real=importlib.metadata.version
def safe_version(name):
    try: return _real(name)
    except importlib.metadata.PackageNotFoundError:
        if name=='eccodes': return 'NOT_INSTALLED_NOT_REQUIRED_FOR_PREPARED_NPZ_REPLAY'
        raise
importlib.metadata.version=safe_version
os.chdir(HERE)
runpy.run_path(str(HERE/'run_stage2_weather_v2.py'),run_name='__main__')
