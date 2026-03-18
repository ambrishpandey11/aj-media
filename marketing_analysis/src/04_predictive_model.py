"""
04_predictive_model.py — Task 4 (Bonus): Predictive ROAS Modelling
Uses XGBoost + feature importance to forecast ROAS and guide budget allocation.
Outputs:
  - outputs/figures/model_feature_importance.png
  - outputs/figures/model_predictions.png
  - outputs/reports/model_results.csv
  - outputs/reports/roas_predictions_by_campaign.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from utils import *

# Optional XGBoost — fall back to GBM if not installed
try:
    from xgboost import XGBRegressor
    USE_XGB = True
except ImportError:
    USE_XGB = False
    print("  [warn] xgboost not installed — using sklearn GradientBoostingRegressor")

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Encode categoricals and create model-ready features."""
    df = df.copy()

    # Lag features (prior-week ROAS by platform)
    df = df.sort_values("Date")
    df["ROAS_lag1w"] = df.groupby("Platform")["ROAS"].shift(7)
    df["Spend_lag1w"] = df.groupby("Platform")["Spend"].shift(7)

    # Rolling 7-day averages
    df["ROAS_roll7"] = df.groupby("Platform")["ROAS"].transform(
        lambda x: x.rolling(7, min_periods=1).mean())

    # Log-transform skewed features
    for col in ["Spend", "Impressions", "Clicks", "CPM", "Customer_LTV"]:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col].clip(0))

    # Encode categoricals
    cat_features = ["Platform", "Region", "Creative_Type", "Product_Category", "Target_Audience"]
    le_map = {}
    for col in cat_features:
        if col in df.columns:
            le = LabelEncoder()
            df[f"enc_{col}"] = le.fit_transform(
                df[col].astype(str).fillna("Unknown"))
            le_map[col] = le

    # Boolean flags
    df["is_video"]      = (df["Creative_Type"].str.lower() == "video").astype(int)
    df["is_search"]     = (df["Creative_Type"].str.lower() == "search").astype(int)
    df["is_comp_event"] = df["Is_Competitive_Event"].astype(int)

    feature_cols = (
        [f"enc_{c}" for c in cat_features] +
        [f"log_{c}" for c in ["Spend","Impressions","Clicks","CPM","Customer_LTV"]] +
        ["CTR","CVR","CPC","Frequency","is_video","is_search","is_comp_event",
         "ROAS_lag1w","Spend_lag1w","ROAS_roll7","Week","Month"] +
        (["Video_Completion_Rate"] if "Video_Completion_Rate" in df.columns else [])
    )
    feature_cols = [c for c in feature_cols if c in df.columns]

    return df, feature_cols

# ─────────────────────────────────────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def train_model(df: pd.DataFrame, feature_cols: list):
    train_df = df.dropna(subset=feature_cols + ["ROAS"])
    X = train_df[feature_cols].fillna(0)
    y = train_df["ROAS"].clip(0, 5)  # cap extreme outliers

    # Winsorise target
    q1, q99 = y.quantile([0.01, 0.99])
    y = y.clip(q1, q99)

    if USE_XGB:
        model = XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1,
            eval_metric="mae"
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42
        )

    # Cross-validated MAE
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring="neg_mean_absolute_error")
    cv_mae = -cv_scores.mean()

    # Fit on full data
    model.fit(X, y)
    y_pred = model.predict(X)
    train_mae = mean_absolute_error(y, y_pred)
    train_r2  = r2_score(y, y_pred)

    print(f"\n  Model performance:")
    print(f"    CV MAE (5-fold):   {cv_mae:.4f}")
    print(f"    Train MAE:         {train_mae:.4f}")
    print(f"    Train R²:          {train_r2:.4f}")
    print(f"    Baseline MAE (mean pred): {mean_absolute_error(y, [y.mean()]*len(y)):.4f}")

    return model, X, y, y_pred, cv_mae, train_r2

# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────
def plot_feature_importance(model, feature_cols: list):
    set_style()
    if hasattr(model, "feature_importances_"):
        importance = pd.Series(model.feature_importances_, index=feature_cols)
    else:
        return

    importance = importance.sort_values(ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = [COLORS["ok"] if imp > importance.median() else COLORS["primary"]
              for imp in importance.values]
    bars = ax.barh(importance.index, importance.values, color=colors, edgecolor="white")
    ax.set_title("Top 15 Features Predicting ROAS\n(XGBoost Feature Importance)",
                 fontweight="bold")
    ax.set_xlabel("Feature Importance Score")
    for bar, val in zip(bars, importance.values):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=8)
    save_fig("model_feature_importance")

def plot_predictions(y_true, y_pred, cv_mae, r2):
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    # Scatter: actual vs predicted
    ax = axes[0]
    ax.scatter(y_true, y_pred, alpha=0.3, s=15, color=COLORS["primary"])
    lim = [min(y_true.min(), y_pred.min()) - 0.1, max(y_true.max(), y_pred.max()) + 0.1]
    ax.plot(lim, lim, "r--", alpha=0.7, label="Perfect prediction")
    ax.set_xlabel("Actual ROAS"); ax.set_ylabel("Predicted ROAS")
    ax.set_title(f"Actual vs Predicted ROAS\nR²={r2:.3f}", fontweight="bold")
    ax.legend(fontsize=8)

    # Residuals
    ax = axes[1]
    residuals = y_pred - y_true
    ax.hist(residuals, bins=50, color=COLORS["primary"], edgecolor="white", alpha=0.85)
    ax.axvline(0, color=COLORS["danger"], linestyle="--")
    ax.set_xlabel("Residual (Predicted − Actual)")
    ax.set_title(f"Residual Distribution\nCV MAE = {cv_mae:.4f}", fontweight="bold")

    # ROAS probability distribution: predicted vs target
    ax = axes[2]
    above_target = (y_pred >= 1.2).mean() * 100
    ax.hist(y_pred, bins=40, color=COLORS["primary"], edgecolor="white", alpha=0.7,
            label="Predicted ROAS dist.")
    ax.axvline(1.2, color=COLORS["danger"], linestyle="--", label=f"Target 1.2")
    ax.axvline(1.4, color=COLORS["ok"],     linestyle="--", label=f"Benchmark 1.4")
    ax.set_xlabel("Predicted ROAS")
    ax.set_title(f"Predicted ROAS Distribution\n{above_target:.1f}% of rows above target",
                 fontweight="bold")
    ax.legend(fontsize=8)

    fig.suptitle("Predictive ROAS Model — Diagnostic Plots", fontweight="bold", fontsize=13)
    save_fig("model_predictions")

def plot_budget_simulation(df: pd.DataFrame, model, feature_cols: list):
    """Simulate ROAS across different spend levels for top platforms."""
    set_style()
    spend_range = np.linspace(df["Spend"].quantile(0.1),
                              df["Spend"].quantile(0.9), 50)

    fig, ax = plt.subplots(figsize=(10, 6))
    for plat in df["Platform"].unique():
        plat_df = df[df["Platform"] == plat].dropna(subset=feature_cols).copy()
        if len(plat_df) < 10: continue
        preds = []
        for spend in spend_range:
            temp = plat_df.copy()
            temp["Spend"] = spend
            temp["log_Spend"] = np.log1p(spend)
            X_temp = temp[feature_cols].fillna(0)
            preds.append(model.predict(X_temp).mean())
        ax.plot(spend_range, preds, linewidth=2,
                color=COLORS.get(plat), label=plat, marker="")

    ax.axhline(1.2, color=COLORS["danger"], linestyle="--", alpha=0.7, label="Target ROAS")
    ax.set_xlabel("Daily Ad Spend ($)")
    ax.set_ylabel("Predicted ROAS")
    ax.set_title("ROAS vs Spend Level by Platform\n(Model Simulation)", fontweight="bold")
    ax.legend()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(fmt_currency))
    save_fig("model_spend_simulation")

# ─────────────────────────────────────────────────────────────────────────────
# CAMPAIGN-LEVEL ROAS FORECASTS
# ─────────────────────────────────────────────────────────────────────────────
def forecast_campaigns(df: pd.DataFrame, model, feature_cols: list):
    campaign_df = df.dropna(subset=feature_cols).copy()
    campaign_df["Predicted_ROAS"] = model.predict(
        campaign_df[feature_cols].fillna(0)).clip(0, 5)

    summary = campaign_df.groupby(["Campaign", "Platform"]).agg(
        Actual_ROAS=("ROAS","mean"),
        Predicted_ROAS=("Predicted_ROAS","mean"),
        Total_Spend=("Spend","sum"),
        Total_Revenue=("Revenue","sum"),
    ).reset_index()
    summary["Budget_Recommendation"] = summary["Predicted_ROAS"].apply(
        lambda r: "Scale Up" if r >= 1.4 else ("Maintain" if r >= 1.2 else "Reduce/Pause"))
    summary = summary.sort_values("Predicted_ROAS", ascending=False)

    summary.to_csv(REPORT_DIR / "roas_predictions_by_campaign.csv", index=False)
    print(f"  [saved] roas_predictions_by_campaign.csv")

    print(f"\n  Campaign recommendations summary:")
    rec_counts = summary["Budget_Recommendation"].value_counts()
    for rec, cnt in rec_counts.items():
        print(f"    {rec:<15}: {cnt} campaigns")

    return summary

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def run():
    print(f"\n{'='*60}")
    print(f"  TASK 4 — PREDICTIVE ROAS MODEL (BONUS)")
    print(f"{'='*60}")

    df = pd.read_csv(REPORT_DIR / "cleaned_data.csv", parse_dates=["Date"])

    print("\n  Engineering features...")
    df_feat, feature_cols = engineer_features(df)
    print(f"    Features: {len(feature_cols)} | Training rows: {df_feat.dropna(subset=feature_cols).shape[0]:,}")

    print("\n  Training model...")
    model, X, y, y_pred, cv_mae, r2 = train_model(df_feat, feature_cols)

    print("\n  Generating visualisations...")
    plot_feature_importance(model, feature_cols)
    plot_predictions(y, y_pred, cv_mae, r2)
    plot_budget_simulation(df_feat, model, feature_cols)

    forecast_df = forecast_campaigns(df_feat, model, feature_cols)

    # Save model metrics
    metrics = pd.DataFrame([{
        "model": "XGBoost" if USE_XGB else "GradientBoosting",
        "cv_mae": cv_mae, "train_r2": r2,
        "n_features": len(feature_cols),
        "n_samples": len(X),
        "pct_above_target_12": (y_pred >= 1.2).mean(),
    }])
    metrics.to_csv(REPORT_DIR / "model_results.csv", index=False)
    print(f"  [saved] model_results.csv")

    print(f"\n  ✓ Task 4 complete.\n")
    return model, forecast_df

if __name__ == "__main__":
    run()
