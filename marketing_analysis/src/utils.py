"""
utils.py — Shared helpers for all analysis scripts
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT / "data"
FIG_DIR     = ROOT / "outputs" / "figures"
REPORT_DIR  = ROOT / "outputs" / "reports"

for d in [FIG_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Brand palette ─────────────────────────────────────────────────────────────
COLORS = {
    "FB":       "#1877F2",
    "Google":   "#EA4335",
    "TT":       "#010101",
    "West":     "#6C63FF",
    "South":    "#FF6584",
    "Northeast":"#43B89C",
    "Midwest":  "#FFC107",
    "primary":  "#6C63FF",
    "danger":   "#FF6584",
    "ok":       "#43B89C",
}

PLATFORM_COLORS  = [COLORS["FB"], COLORS["Google"], COLORS["TT"]]
REGION_COLORS    = [COLORS["West"], COLORS["South"], COLORS["Northeast"], COLORS["Midwest"]]

def set_style():
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update({
        "figure.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
    })

def save_fig(name: str, tight=True):
    path = FIG_DIR / f"{name}.png"
    if tight:
        plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {path.name}")

def load_data(filename: str = None) -> pd.DataFrame:
    """Auto-detect and load the Excel dataset from data/"""
    if filename:
        path = DATA_DIR / filename
    else:
        xlsx_files = list(DATA_DIR.glob("*.xlsx")) + list(DATA_DIR.glob("*.xls"))
        if not xlsx_files:
            raise FileNotFoundError(
                f"No Excel file found in {DATA_DIR}. "
                "Please place your marketing_data.xlsx there."
            )
        path = xlsx_files[0]
        print(f"  [load] Reading: {path.name}")
    return pd.read_excel(path)

def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate CTR, CPC, CVR, ROAS and add to dataframe."""
    df = df.copy()
    df["CTR"]  = np.where(df["Impressions"] > 0, df["Clicks"] / df["Impressions"], np.nan)
    df["CPC"]  = np.where(df["Clicks"] > 0,      df["Spend"] / df["Clicks"],       np.nan)
    df["CVR"]  = np.where(df["Clicks"] > 0,      df["Purchases"] / df["Clicks"],   np.nan)
    df["ROAS"] = np.where(df["Spend"] > 0,       df["Revenue"] / df["Spend"],      np.nan)
    return df

def roas_color(val: float) -> str:
    if val >= 1.4: return COLORS["ok"]
    if val >= 1.2: return COLORS["primary"]
    return COLORS["danger"]

def fmt_currency(x, pos=None):
    if x >= 1_000_000: return f"${x/1e6:.1f}M"
    if x >= 1_000:     return f"${x/1e3:.0f}K"
    return f"${x:.0f}"
