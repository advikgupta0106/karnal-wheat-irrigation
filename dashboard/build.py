"""Build dashboard/index.html: embeds data/karnal_daily.csv into template.html.

Run from the project root:  python dashboard/build.py
Then open dashboard/index.html in a browser (no server needed).
"""
import csv
import json
from datetime import datetime
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
# Which product each baked row came from (era5 / forecast-analysis). The page appends
# "live" for anything it tops up itself, so all three stay distinguishable in the browser.
data["source"] = [r.get("source", "era5") for r in rows]

# The as-of date is the last day in the data. It is injected, never typed by hand,
# so the banner and footer can never drift from what the page actually replays.
asof = datetime.strptime(data["dates"][-1], "%Y-%m-%d").strftime("%d %b %Y").lstrip("0")

html = (Path(__file__).parent / "template.html").read_text()
html = html.replace("__DATA__", json.dumps(data, separators=(",", ":")))
html = html.replace("__ASOF__", asof)

# The fitted Step 4 model, exported by the last cell of wheat.ipynb. Embedded rather than
# fetched, so the page stays a single file that opens from disk with no server.
model = json.loads((Path(__file__).parent / "model.json").read_text())
assert len(model["features"]) == len(model["coef"]) == len(model["mean"]) == len(model["scale"]), \
    "model.json is inconsistent: features, coef, mean and scale must be the same length"
html = html.replace("__MODEL__", json.dumps(model, separators=(",", ":")))
for ph in ("__DATA__", "__ASOF__", "__MODEL__"):
    assert ph not in html, f"placeholder {ph} was left unfilled"
(Path(__file__).parent / "index.html").write_text(html)
print(f"Wrote dashboard/index.html ({len(rows)} days, as of {asof}; "
      f"model: {len(model['features'])} features, test MAE {model['test_mae']})")
