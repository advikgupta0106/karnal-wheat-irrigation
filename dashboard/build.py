"""Build dashboard/index.html: embeds data/karnal_daily.csv and the notebook's ET0 model into template.html.

Run from the project root:  python dashboard/build.py
Then open dashboard/index.html in a browser (no server needed).

The model is the notebook's best one (wheat.ipynb, Cells 47-52): linear regression on 11
features ("Linear regression (11 features, with lags)"), trained on seasons 2018-19 to 2023-24
and tested on 2024-25 and 2025-26. It is refitted here in plain Python (no numpy/sklearn needed)
and the build stops if its test scores do not match the notebook's published numbers.
"""
import csv
import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # project root, so the script works from any folder

# Short names used in the browser -> column names in data/karnal_daily.csv (Open-Meteo variables).
# The NASA POWER cross-check columns in the CSV are not used by the dashboard.
COLS = {
    "et0": "et0_fao_evapotranspiration", "rain": "precipitation_sum",
    "tmax": "temperature_2m_max", "tmin": "temperature_2m_min",
    "rh": "relative_humidity_2m_mean", "wind": "wind_speed_10m_max",
    "rad": "shortwave_radiation_sum",
}
# The notebook's split (Cell 9): whole seasons, never shuffled, so every training day is before every test day.
TRAIN_SEASONS = ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24"]
TEST_SEASONS = ["2024-25", "2025-26"]
# Same order as the notebook's FEATURES_LAG. Keys refer to the JS-side names.
FEATURES = ["et0", "tmax", "tmin", "wind", "rad", "rh", "rain", "dos", "et0_yesterday", "et0_3day_mean", "et0_change"]
# et0_change == et0 - et0_yesterday exactly, so it is collinear. sklearn's least-squares solve gives
# the same predictions with or without it; we drop it from the solve (its weight is 0).
SOLVE = [f for f in FEATURES if f != "et0_change"]
NOTEBOOK_MAE = {"model": 0.2980, "persistence": 0.3230}  # Cell 31 / Cell 38 of wheat.ipynb
RAIN_THRESHOLD = 1.0  # mm, the notebook's rainy-day definition (Cell 29)


def season(d):
    """Name of the Nov-Mar wheat season a date belongs to, e.g. '2025-26', or None for Apr-Oct.

    Nov and Dec start a season; Jan-Mar belong to the season that started the previous November.
    Same rule as the notebook's wheat_season() (Cell 8).
    """
    if d.month >= 11:
        return f"{d.year}-{str(d.year + 1)[2:]}"
    if d.month <= 3:
        return f"{d.year - 1}-{str(d.year)[2:]}"
    return None


def solve(A, b):
    """Solve A x = b for x (A is a small square matrix, as lists of lists).

    Gaussian elimination with partial pivoting: this stands in for numpy.linalg.solve so the build needs
    no packages. Only ever called on a 10x10 system, so speed does not matter.
    """
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]  # augmented matrix [A | b], copied so A is untouched
    for c in range(n):
        # Swap in the row with the largest value in this column, to avoid dividing by a tiny number.
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        # Subtract multiples of row c to make every entry below M[c][c] zero.
        for r in range(c + 1, n):
            f = M[r][c] / M[c][c]
            for k in range(c, n + 1):
                M[r][k] -= f * M[c][k]
    # Back-substitution: the matrix is now upper-triangular, so solve from the last row upward.
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (M[r][n] - sum(M[r][k] * x[k] for k in range(r + 1, n))) / M[r][r]
    return x


# ---- Load the daily record into the shape the browser expects ----------------------------
# One list per variable, all aligned by position: data["et0"][i] is the ET0 on data["dates"][i].
with open(ROOT / "data" / "karnal_daily.csv", newline="") as fh:
    rows = list(csv.DictReader(fh))
data = {"dates": [r["date"] for r in rows]}
for key, col in COLS.items():
    data[key] = [float(r[col]) for r in rows]

# ---- Rebuild the notebook's modelling table (Cells 8 and 22) ----------------------------
# One example per day i: the features are what is known on day i, the target is day i+1's ET0.
# The CSV is one continuous run of days with no gaps (checked in the notebook's Cell 7),
# so position i-1 really is yesterday and i+1 really is tomorrow.
dates = [date.fromisoformat(d) for d in data["dates"]]
et0, rain = data["et0"], data["rain"]
examples = []  # (season, features dict, target, today's et0, tomorrow's rain)
for i in range(2, len(rows) - 1):  # start at 2 so the 3-day mean has two earlier days; stop before the last day (no tomorrow)
    s = season(dates[i])
    if s is None or s != season(dates[i + 1]):
        continue  # outside Nov-Mar, or the last day of a season (31 Mar -> 1 Apr is not a wheat-season pair)
    # Today's seven weather values, then day of season and the three lag features.
    f = {k: data[k][i] for k in ["et0", "tmax", "tmin", "wind", "rad", "rh", "rain"]}
    f["dos"] = (dates[i] - date(int(s[:4]), 11, 1)).days
    f["et0_yesterday"] = et0[i - 1]
    f["et0_3day_mean"] = (et0[i] + et0[i - 1] + et0[i - 2]) / 3
    f["et0_change"] = et0[i] - et0[i - 1]
    # Today's ET0 is kept to score the "tomorrow = today" baseline; tomorrow's rain to split rainy and dry test days.
    examples.append((s, f, et0[i + 1], et0[i], rain[i + 1]))

train = [e for e in examples if e[0] in TRAIN_SEASONS]
test = [e for e in examples if e[0] in TEST_SEASONS]

# ---- Fit OLS on standardised features (as the notebook's StandardScaler + LinearRegression) ----
# 1. Standardise each feature with the TRAINING mean and standard deviation (population std, like
#    StandardScaler), so features in mm, degC, % and days are on one scale and the solve is well-conditioned.
n = len(train)
mean = {k: sum(e[1][k] for e in train) / n for k in SOLVE}
std = {k: (sum((e[1][k] - mean[k]) ** 2 for e in train) / n) ** 0.5 for k in SOLVE}
ybar = sum(e[2] for e in train) / n
Z = [[(e[1][k] - mean[k]) / std[k] for k in SOLVE] for e in train]
yc = [e[2] - ybar for e in train]  # centred target: with centred features the intercept is just ybar
# 2. Least squares via the normal equations: the weights beta solve (Z^T Z) beta = Z^T y.
ZtZ = [[sum(z[a] * z[b] for z in Z) for b in range(len(SOLVE))] for a in range(len(SOLVE))]
Zty = [sum(z[a] * y for z, y in zip(Z, yc)) for a in range(len(SOLVE))]
beta = solve(ZtZ, Zty)

# 3. Convert to raw-unit weights so the browser can skip the scaling and compute
#    ET0_tomorrow = b + sum(w_k * x_k).  Since beta_k * (x_k - mean_k) / std_k = (beta_k / std_k) * x_k - ...,
#    each weight is beta_k / std_k and the mean terms fold into the intercept.
weights = {k: 0.0 for k in FEATURES}
for k, bk in zip(SOLVE, beta):
    weights[k] = bk / std[k]
intercept = ybar - sum(weights[k] * mean[k] for k in SOLVE)


def predict(f):
    """Tomorrow's ET0 from one day's features. The same formula the browser runs (template.html, model())."""
    return intercept + sum(weights[k] * f[k] for k in FEATURES)


# ---- Score on the test seasons and check against the notebook ----------------------------
# If these numbers drift from the notebook's (data changed, a feature built differently...), the build
# fails, so the app can never quietly show a model that is not the one the notebook evaluated.
def mae(pairs):
    """Mean absolute error of (prediction, actual) pairs, in mm/day."""
    return sum(abs(p - y) for p, y in pairs) / len(pairs)


mae_model = mae([(predict(e[1]), e[2]) for e in test])
mae_pers = mae([(e[3], e[2]) for e in test])
rainy = [e for e in test if e[4] > RAIN_THRESHOLD]
dry = [e for e in test if e[4] <= RAIN_THRESHOLD]
for name, got in [("model", mae_model), ("persistence", mae_pers)]:
    assert abs(got - NOTEBOOK_MAE[name]) < 5e-4, (
        f"{name} test MAE {got:.4f} does not match the notebook's {NOTEBOOK_MAE[name]:.4f}")

# Everything the app needs about the model: the weights to run it, and the scores it shows on the
# Settings screen and in the rain warning (rainy = tomorrow had more than 1 mm, as in the notebook).
data["model"] = {
    "name": "Linear regression (11 features, with lags)",
    "w": {k: round(v, 8) for k, v in weights.items()},
    "b": round(intercept, 8),
    "train": f"{TRAIN_SEASONS[0]} to {TRAIN_SEASONS[-1]}",
    "test": " and ".join(TEST_SEASONS),
    "n_test": len(test),
    "mae": round(mae_model, 4),
    "mae_pers": round(mae_pers, 4),
    "skill": round((mae_pers - mae_model) / mae_pers * 100, 2),
    "mae_dry": round(mae([(predict(e[1]), e[2]) for e in dry]), 4),
    "mae_rainy": round(mae([(predict(e[1]), e[2]) for e in rainy]), 4),
    "bias_rainy": round(sum(predict(e[1]) - e[2] for e in rainy) / len(rainy), 3),
}

# ---- Write the page: put the data into the template as a compact JSON literal --------------
html = (Path(__file__).parent / "template.html").read_text().replace(
    "__DATA__", json.dumps(data, separators=(",", ":")))
(Path(__file__).parent / "index.html").write_text(html)
m = data["model"]
print(f"Model: test MAE {m['mae']} vs persistence {m['mae_pers']} ({m['skill']:+.2f}%), "
      f"matches notebook. Rainy-day MAE {m['mae_rainy']}, bias {m['bias_rainy']:+}")
print(f"Wrote dashboard/index.html ({len(rows)} days, to {data['dates'][-1]})")
