"""
train_model.py
--------------
Trains and compares Linear Regression, Random Forest and Gradient Boosting for
monthly cloud-cost prediction. Evaluates on a held-out 20% test set and with
five-fold cross-validation, saves the best pipeline, and writes the figures used
in the Results chapter.
"""

import json
import datetime as dt
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

plt.rcParams.update({"figure.dpi": 130, "font.size": 11})

# Project-relative paths
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "cloud_costs.csv"
FIG = ROOT / "figures"
ML_DIR = ROOT / "ml"

FIG.mkdir(parents=True, exist_ok=True)
ML_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)
y = df["monthly_cost"].values
X = df.drop(columns=["monthly_cost"])

cat_cols = ["provider", "region", "instance_type"]
num_cols = ["vcpu", "ram_gb", "instance_count", "monthly_hours",
            "storage_gb", "egress_gb", "managed_db", "load_balancer", "avg_cpu_util"]

pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", StandardScaler(), num_cols),
])

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=300, max_depth=None, min_samples_leaf=2,
        random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=500, max_depth=3, learning_rate=0.08,
        subsample=0.9, random_state=42),
}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)

results = {}
fitted = {}
for name, est in models.items():
    pipe = Pipeline([("pre", pre), ("model", est)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    r2 = r2_score(y_test, pred)
    cv = cross_val_score(pipe, X, y, cv=5, scoring="r2", n_jobs=-1)
    results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2,
                     "CV_R2_mean": float(cv.mean()), "CV_R2_sd": float(cv.std())}
    fitted[name] = pipe
    print(f"{name:20s}  MAE {mae:8.2f}  RMSE {rmse:8.2f}  R2 {r2:.4f}  "
          f"CV {cv.mean():.4f} +/- {cv.std():.4f}")

# Best model by test R2
best_name = max(results, key=lambda k: results[k]["R2"])
best = fitted[best_name]
joblib.dump(best, ML_DIR / "model.joblib")
joblib.dump(fitted["Random Forest"], ML_DIR / "model_rf.joblib")  # used by /api/forecast's confidence bands
with open(ML_DIR / "metrics.json", "w") as f:
    json.dump({
        "results": results, "best": best_name,
        "mean_cost": round(float(y.mean()), 2), "sample_size": int(len(df)),
        "cv_folds": 5, "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "trigger": "offline-train",
    }, f, indent=2)
print(f"\nBest model: {best_name}")

# ---------------------------------------------------------------------------
# Figure: model comparison (MAE, RMSE, R2)
# ---------------------------------------------------------------------------
names = list(results.keys())
colors = ["#6b7280", "#2563eb", "#16a34a"]

fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
for ax, metric, title in zip(
        axes, ["MAE", "RMSE", "R2"],
        ["Mean Absolute Error ($)", "Root Mean Sq. Error ($)", "R\u00b2 (test set)"]):
    vals = [results[n][metric] for n in names]
    bars = ax.bar(range(len(names)), vals, color=colors)
    ax.set_title(title)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v, f"{v:.3f}" if metric == "R2" else f"{v:.1f}",
                ha="center", va="bottom", fontsize=9)
    ax.margins(y=0.15)
fig.tight_layout()
fig.savefig(FIG / "model_comparison.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure: predicted vs actual for the best model
# ---------------------------------------------------------------------------
pred_best = best.predict(X_test)
fig, ax = plt.subplots(figsize=(5.4, 5.2))
ax.scatter(y_test, pred_best, s=8, alpha=0.35, color="#16a34a", edgecolor="none")
lim = [0, max(y_test.max(), pred_best.max())*1.02]
ax.plot(lim, lim, "--", color="#111827", lw=1)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("Actual monthly cost ($)")
ax.set_ylabel("Predicted monthly cost ($)")
ax.set_title(f"Predicted vs Actual \u2014 {best_name}\nR\u00b2 = {results[best_name]['R2']:.3f}")
fig.tight_layout()
fig.savefig(FIG / "pred_vs_actual.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure: feature importance for the best tree model
# ---------------------------------------------------------------------------
ohe = best.named_steps["pre"].named_transformers_["cat"]
feat_names = list(ohe.get_feature_names_out(cat_cols)) + num_cols
importances = best.named_steps["model"].feature_importances_
imp = pd.Series(importances, index=feat_names).sort_values(ascending=False).head(12)

fig, ax = plt.subplots(figsize=(7.2, 4.6))
imp[::-1].plot.barh(ax=ax, color="#16a34a")
ax.set_title(f"Top feature importances \u2014 {best_name}")
ax.set_xlabel("Relative importance")
fig.tight_layout()
fig.savefig(FIG / "feature_importance.png", bbox_inches="tight")
plt.close(fig)

print("Saved figures: model_comparison, pred_vs_actual, feature_importance")
