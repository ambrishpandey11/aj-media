"""
01_data_quality.py — Task 1: Data Quality Assessment & Cleaning
Outputs:
  - outputs/reports/data_quality_report.csv   (cleaning log)
  - outputs/reports/cleaned_data.csv          (clean dataset used by all subsequent scripts)
  - outputs/figures/missing_values_heatmap.png
  - outputs/figures/outlier_boxplots.png
  - outputs/figures/roas_distribution.png
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from utils import *

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────────────────────────────────────
def load_and_inspect(df_raw: pd.DataFrame) -> dict:
    """Return a dict of quality issues found."""
    issues = {}

    # Shape
    issues["shape"] = df_raw.shape
    print(f"\n{'='*60}")
    print(f"  TASK 1 — DATA QUALITY ASSESSMENT")
    print(f"{'='*60}")
    print(f"  Raw shape: {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")

    # Column name normalisation map
    rename = {}
    for c in df_raw.columns:
        clean = c.strip().title().replace("_", " ")
        if c != clean:
            rename[c] = clean
    if rename:
        issues["renamed_columns"] = rename

    # Missing values
    missing = df_raw.isnull().sum()
    missing_pct = (missing / len(df_raw) * 100).round(2)
    missing_df = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
    missing_df = missing_df[missing_df["missing_count"] > 0].sort_values("missing_count", ascending=False)
    issues["missing"] = missing_df
    print(f"\n  Missing values:")
    if len(missing_df) == 0:
        print("    None found ✓")
    else:
        print(missing_df.to_string())

    # Duplicates
    dup_count = df_raw.duplicated().sum()
    issues["duplicates"] = dup_count
    print(f"\n  Duplicate rows: {dup_count}")

    # Data types
    issues["dtypes"] = df_raw.dtypes

    return issues

# ─────────────────────────────────────────────────────────────────────────────
# 2. CLEAN
# ─────────────────────────────────────────────────────────────────────────────
def clean(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Apply all cleaning steps. Returns (clean_df, cleaning_log)."""
    log = []
    df  = df_raw.copy()

    # ── Column names ──────────────────────────────────────────────────────────
    df.columns = df.columns.str.strip()

    # Standardise common variants — map to actual column names in the dataset
    col_map = {
        "ad_spend":      "Spend",
        "adspend":       "Spend",
        "spend":         "Spend",
        "impressions":   "Impressions",
        "clicks":        "Clicks",
        "purchases":     "Purchases",
        "revenue":       "Revenue",
        "frequency":     "Frequency",
        "cpm":           "CPM",
        "date":          "Date",
        "platform":      "Platform",
        "campaign":      "Campaign",
        "campaign_name": "Campaign",
        "region":        "Region",
        "product_category": "Product_Category",
        "target_audience":  "Target_Audience",
        "creative_type":    "Creative_Type",
        "video_completion_rate": "Video_Completion_Rate",
        "customer_ltv":  "Customer_LTV",
        "is_competitive_event": "Is_Competitive_Event",
    }
    df.rename(columns={c: col_map[c.lower().replace(" ", "_")]
                       for c in df.columns
                       if c.lower().replace(" ", "_") in col_map}, inplace=True)

    # ── Handle Impressions == 0 ──────────────────────────────────────────────
    # We DO NOT zero out Revenue or Purchases here. In marketing, 0 impressions
    # with >0 revenue is often valid (e.g. delayed attribution from a previous day).
    # Zeroing it out destroys real revenue and lowers the ROAS artificially.
    if "Impressions" in df.columns:
        zero_imp_mask = df["Impressions"] == 0
        n_zero_imp = zero_imp_mask.sum()
        if n_zero_imp > 0:
            # Only zero out Clicks if Impressions are 0 (you can't click what you didn't see today)
            df.loc[zero_imp_mask, "Clicks"] = 0
            log.append({"step": "zero_impressions", "action": f"Set Clicks=0 for {n_zero_imp} rows with Impressions==0 (Preserved Revenue/Purchases)"})

    # ── Date parsing ──────────────────────────────────────────────────────────
    if "Date" in df.columns:
        before = df["Date"].dtype
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        bad_dates = df["Date"].isna().sum()
        if bad_dates:
            df.dropna(subset=["Date"], inplace=True)
            log.append({"step": "date_parse", "action": f"Dropped {bad_dates} rows with unparseable dates"})
        df["Week"]  = df["Date"].dt.isocalendar().week.astype(int)
        df["Month"] = df["Date"].dt.month

    # ── Numeric coercion ──────────────────────────────────────────────────────
    numeric_cols = ["Spend", "CPM", "Impressions", "Frequency",
                    "Clicks", "Purchases", "Revenue",
                    "Video_Completion_Rate", "Customer_LTV"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Negative values → NaN ─────────────────────────────────────────────────
    for col in ["Spend", "Impressions", "Clicks", "Purchases", "Revenue", "CPM"]:
        if col in df.columns:
            neg = (df[col] < 0).sum()
            if neg:
                df.loc[df[col] < 0, col] = np.nan
                log.append({"step": "negative_values", "action": f"Set {neg} negative {col} → NaN"})

    # ── Video Completion Rate: must be 0–1 (or 0–100 → convert) ──────────────
    if "Video_Completion_Rate" in df.columns:
        if df["Video_Completion_Rate"].max() > 1.5:
            df["Video_Completion_Rate"] = df["Video_Completion_Rate"] / 100
            log.append({"step": "vcr_scale", "action": "Video_Completion_Rate divided by 100 (was in %, converted to 0–1)"})
        df["Video_Completion_Rate"] = df["Video_Completion_Rate"].clip(0, 1)

    # ── Is_Competitive_Event → boolean ───────────────────────────────────────
    if "Is_Competitive_Event" in df.columns:
        df["Is_Competitive_Event"] = df["Is_Competitive_Event"].fillna(0).astype(bool)

    # ── Categorical columns: strip whitespace & title-case ───────────────────
    cat_cols = ["Platform", "Campaign", "Region",
                "Product_Category", "Target_Audience", "Creative_Type"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()
            df[col] = df[col].replace("Nan", np.nan)

    # ── Impute missing numerics ─────────────────────────────────────────────
    # Revenue: missing = no sale occurred → fill with 0 (not median)
    if "Revenue" in df.columns:
        n_miss_rev = df["Revenue"].isna().sum()
        if n_miss_rev:
            df["Revenue"] = df["Revenue"].fillna(0)
            log.append({"step": "impute_revenue", "action": f"Filled {n_miss_rev} missing Revenue with 0 (no sale)"})

    # Video_Completion_Rate: NaN = non-video creative → leave as NaN (not missing data)
    # Do NOT impute VCR — it is only populated for Video creative rows

    # Other numeric columns: impute with median
    cols_to_impute = [c for c in numeric_cols if c not in ("Revenue", "Video_Completion_Rate")]
    for col in cols_to_impute:
        if col in df.columns:
            n_missing = df[col].isna().sum()
            if n_missing:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                log.append({"step": "impute_median", "action": f"Filled {n_missing} missing {col} with median ({median_val:.2f})"})

    # ── Remove duplicates ──────────────────────────────────────────────────────
    n_dups = df.duplicated().sum()
    if n_dups:
        df.drop_duplicates(inplace=True)
        log.append({"step": "dedup", "action": f"Removed {n_dups} duplicate rows"})

    # ── Outlier treatment (percentile-based winsorize: 1st/99th) ──────────────
    # IMPORTANT: Do NOT winsorise Spend or Revenue — they are the numerator
    # and denominator of ROAS. Clipping them artificially inflates aggregate ROAS.
    # Only winsorise ancillary metrics that don't directly compose the target KPI.
    for col in ["CPM", "Customer_LTV"]:
        if col in df.columns:
            lo = df[col].quantile(0.01)
            hi = df[col].quantile(0.99)
            outliers = ((df[col] < lo) | (df[col] > hi)).sum()
            if outliers:
                df[col] = df[col].clip(lo, hi)
                log.append({"step": "winsorise", "action": f"Winsorised {outliers} outliers in {col} [1st/99th percentile]"})

    # ── Derived metrics ───────────────────────────────────────────────────────
    df = add_derived_metrics(df)
    log.append({"step": "derived_metrics", "action": "Added CTR, CPC, CVR, ROAS columns"})

    print(f"\n  Cleaning steps applied: {len(log)}")
    for entry in log:
        print(f"    [{entry['step']}] {entry['action']}")

    print(f"\n  Clean shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df, log

# ─────────────────────────────────────────────────────────────────────────────
# 3. VALIDATE METRICS
# ─────────────────────────────────────────────────────────────────────────────
def validate_metrics(df: pd.DataFrame):
    print(f"\n  Key metric validation:")
    # Use aggregate ROAS (total revenue / total spend) — NOT mean of per-row ROAS
    # The mean of per-row ROAS is skewed upward by low-spend rows with high ROAS
    agg_roas = df["Revenue"].sum() / df["Spend"].sum() if df["Spend"].sum() > 0 else 0
    metrics = {
        "Overall ROAS":       agg_roas,
        "Mean CTR":           df["CTR"].mean() * 100,
        "Mean CVR":           df["CVR"].mean() * 100,
        "Mean CPC ($)":       df["CPC"].mean(),
        "Total Spend ($)":    df["Spend"].sum(),
        "Total Revenue ($)":  df["Revenue"].sum(),
        "Total Purchases":    df["Purchases"].sum(),
    }
    for k, v in metrics.items():
        flag = ""
        if "ROAS" in k and v < 1.2: flag = "  ⚠ BELOW TARGET"
        print(f"    {k:<25} {v:>12,.2f}{flag}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────
def plot_missing_heatmap(df_raw: pd.DataFrame):
    set_style()
    missing_pct = df_raw.isnull().mean() * 100
    missing_pct = missing_pct[missing_pct > 0]
    if len(missing_pct) == 0:
        print("  [skip] No missing values — heatmap not needed")
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.barh(missing_pct.index, missing_pct.values,
                   color=[COLORS["danger"] if v > 20 else COLORS["primary"] for v in missing_pct.values])
    ax.set_xlabel("Missing %")
    ax.set_title("Missing Values by Column", fontweight="bold")
    ax.axvline(5, color="gray", linestyle="--", alpha=0.5, label="5% threshold")
    for bar, val in zip(bars, missing_pct.values):
        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=9)
    ax.legend()
    save_fig("missing_values_heatmap")

def plot_outlier_boxplots(df: pd.DataFrame):
    set_style()
    cols = [c for c in ["Spend", "Revenue", "CPM", "Customer_LTV", "ROAS"] if c in df.columns]
    fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 5))
    if len(cols) == 1: axes = [axes]
    for ax, col in zip(axes, cols):
        ax.boxplot(df[col].dropna(), patch_artist=True,
                   boxprops=dict(facecolor=COLORS["primary"], alpha=0.6),
                   medianprops=dict(color="white", linewidth=2))
        ax.set_title(col, fontweight="bold")
        ax.set_xticks([])
    fig.suptitle("Distribution of Key Numeric Metrics (after cleaning)", fontweight="bold", y=1.01)
    save_fig("outlier_boxplots")

def plot_roas_distribution(df: pd.DataFrame):
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Overall ROAS distribution
    ax = axes[0]
    ax.hist(df["ROAS"].dropna(), bins=40, color=COLORS["primary"], edgecolor="white", alpha=0.85)
    ax.axvline(1.2, color=COLORS["danger"],  linestyle="--", label="Target ROAS (1.2)")
    ax.axvline(1.4, color=COLORS["ok"],      linestyle="--", label="Benchmark (1.4)")
    ax.axvline(df["ROAS"].mean(), color="black", linestyle="-", label=f"Mean ({df['ROAS'].mean():.2f})")
    ax.set_title("Overall ROAS Distribution", fontweight="bold")
    ax.set_xlabel("ROAS"); ax.set_ylabel("Count")
    ax.legend(fontsize=8)

    # ROAS by platform
    ax = axes[1]
    platforms = df.groupby("Platform")["ROAS"].mean().sort_values()
    colors_p  = [roas_color(v) for v in platforms.values]
    bars = ax.barh(platforms.index, platforms.values, color=colors_p)
    ax.axvline(1.2, color=COLORS["danger"], linestyle="--", alpha=0.7, label="Target 1.2")
    ax.set_title("Mean ROAS by Platform", fontweight="bold")
    ax.set_xlabel("ROAS")
    for bar, val in zip(bars, platforms.values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=9)
    ax.legend(fontsize=8)

    # ROAS over time
    ax = axes[2]
    weekly = df.groupby("Week")["ROAS"].mean()
    ax.plot(weekly.index, weekly.values, marker="o", color=COLORS["primary"], linewidth=2)
    ax.axhline(1.2, color=COLORS["danger"], linestyle="--", alpha=0.7, label="Target 1.2")
    ax.fill_between(weekly.index, weekly.values, 1.2,
                    where=(weekly.values < 1.2), alpha=0.15, color=COLORS["danger"])
    ax.set_title("Weekly ROAS Trend", fontweight="bold")
    ax.set_xlabel("Week"); ax.set_ylabel("ROAS")
    ax.legend(fontsize=8)

    fig.suptitle("ROAS Overview — Data Quality Check", fontweight="bold", fontsize=13)
    save_fig("roas_distribution")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def run():
    df_raw = load_data()
    issues = load_and_inspect(df_raw)
    df, log = clean(df_raw)
    validate_metrics(df)

    # Plots
    print("\n  Generating visualisations...")
    plot_missing_heatmap(df_raw)
    plot_outlier_boxplots(df)
    plot_roas_distribution(df)

    # Save cleaning log
    log_df = pd.DataFrame(log)
    log_df.to_csv(REPORT_DIR / "data_quality_report.csv", index=False)
    print(f"  [saved] data_quality_report.csv")

    # Save clean dataset for downstream scripts
    df.to_csv(REPORT_DIR / "cleaned_data.csv", index=False)
    print(f"  [saved] cleaned_data.csv")

    print(f"\n  ✓ Task 1 complete.\n")
    return df

if __name__ == "__main__":
    run()
