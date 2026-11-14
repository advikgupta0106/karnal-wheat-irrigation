# Wheat Irrigation Decision Tool — Karnal, Haryana

Predicts tomorrow's crop water loss (reference evapotranspiration, ET0) from today's
weather and turns that into a simple irrigate / don't-irrigate recommendation for
wheat growers near Karnal, Haryana (29.69 N, 76.99 E).

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook wheat.ipynb
```

Run the notebook top to bottom. The download cells hit two keyless public APIs and
write a merged daily table to `data/karnal_daily.csv`.

## Sources

- **Open-Meteo Historical Weather API** (ERA5 reanalysis) — primary source.
  https://open-meteo.com/en/docs/historical-weather-api
- **NASA POWER daily point API** (community=AG) — independent cross-check.
  https://power.larc.nasa.gov/docs/services/api/temporal/daily/

Note: ET0 in this project is **calculated** from weather variables (FAO-56
Penman-Monteith), not measured in a field. That is a limitation we state in the
write-up.
