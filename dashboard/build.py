"""Build dashboard/index.html: embeds data/karnal_daily.csv into template.html.

Run from the project root:  python dashboard/build.py
Then open dashboard/index.html in a browser (no server needed).
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COLS = {
    "et0": "et0_fao_evapotranspiration", "rain": "precipitation_sum",
    "tmax": "temperature_2m_max", "tmin": "temperature_2m_min",
    "rh": "relative_humidity_2m_mean", "wind": "wind_speed_10m_max",
    "rad": "shortwave_radiation_sum",
}

rows = list(csv.DictReader(open(ROOT / "data" / "karnal_daily.csv")))
data = {"dates": [r["date"] for r in rows]}
for key, col in COLS.items():
    data[key] = [float(r[col]) for r in rows]

html = (Path(__file__).parent / "template.html").read_text().replace(
    "__DATA__", json.dumps(data, separators=(",", ":")))
(Path(__file__).parent / "index.html").write_text(html)
print(f"Wrote dashboard/index.html ({len(rows)} days, to {data['dates'][-1]})")
