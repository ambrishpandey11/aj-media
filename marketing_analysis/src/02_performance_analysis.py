"""
02_performance_analysis.py — Task 2: Comprehensive Performance Analysis
Outputs:
  - outputs/figures/channel_performance.png
  - outputs/figures/regional_performance.png
  - outputs/figures/creative_performance.png
  - outputs/figures/product_audience_performance.png
  - outputs/figures/wow_trends.png
  - outputs/figures/competitive_events.png
  - outputs/figures/frequency_vs_cvr.png
  - outputs/reports/performance_summary.csv
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
# HELPER: stat significance (Kruskal-Wallis for multi-group, Mann-Whitney for 2)
# ─────────────────────────────────────────────────────────────────────────────
def kruskal_significance(df: pd.DataFrame, group_col: str, metric: str) -> str:
    groups = [g[metric].dropna().values for _, g in df.groupby(group_col)]
    if len(groups) < 2 or any(len(g) < 3 for g in groups):
        return "n/a"
    stat, p = stats.kruskal(*groups)
    if p < 0.001: return f"p<0.001 ***"
    if p < 0.01:  return f"p={p:.3f} **"
    if p < 0.05:  return f"p={p:.3f} *"
    return f"p={p:.3f} (ns)"

def mannwhitney(a, b) -> str:
    if len(a) < 3 or len(b) < 3: return "n/a"
    _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return f"p={p:.3f}" + (" ***" if p<0.001 else " **" if p<0.01 else " *" if p<0.05 else " (ns)")

# ─────────────────────────────────────────────────────────────────────────────
# 1. CHANNEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
def channel_analysis(df: pd.DataFrame):
    print("\n  [1/7] Channel performance analysis...")
    set_style()

    metrics = ["ROAS", "CTR", "CVR", "CPC"]
    summary = df.groupby("Platform").agg(
        Total_Spend=("Spend", "sum"),
        Revenue=("Revenue", "sum"),
        Purchases=("Purchases", "sum"),
        ROAS=("ROAS", "mean"),
        CTR=("CTR", "mean"),
        CVR=("CVR", "mean"),
        CPC=("CPC", "mean"),
        CPM=("CPM", "mean"),
    ).reset_index()

    print(summary.to_string(index=False))
    for m in metrics:
        sig = kruskal_significance(df, "Platform", m)
        print(f"    Kruskal-Wallis {m}: {sig}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    platforms = summary["Platform"].tolist()
    colors    = [COLORS.get(p, COLORS["primary"]) for p in platforms]

    for ax, metric in zip(axes, metrics):
        vals = summary[metric].values
        bars = ax.bar(platforms, vals, color=colors, edgecolor="white", linewidth=0.8)
        ax.set_title(f"{metric} by Platform\n({kruskal_significance(df, 'Platform', metric)})",
                     fontweight="bold", fontsize=10)
        ax.set_ylabel(metric)
        if metric == "ROAS":
            ax.axhline(1.2, color=COLORS["danger"], linestyle="--", alpha=0.7, label="Target 1.2")
            ax.legend(fontsize=8)
        for bar, val in zip(bars, vals):
            fmt = f"{val:.2%}" if metric in ["CTR", "CVR"] else f"{val:.2f}"
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01,
                    fmt, ha="center", fontsize=9, fontweight="bold")

    fig.suptitle("Channel Performance Analysis — FB vs Google vs TT",
                 fontweight="bold", fontsize=13)
    save_fig("channel_performance")

    # Spend vs Revenue bubble
    fig, ax = plt.subplots(figsize=(8, 6))
    for _, row in summary.iterrows():
        c = COLORS.get(row["Platform"], COLORS["primary"])
        ax.scatter(row["Total_Spend"]/1e6, row["Revenue"]/1e6,
                   s=row["Purchases"]/10, color=c, alpha=0.8, label=row["Platform"], zorder=3)
        ax.annotate(row["Platform"], (row["Total_Spend"]/1e6, row["Revenue"]/1e6),
                    textcoords="offset points", xytext=(8, 5), fontsize=10)

    spend_max = summary["Total_Spend"].max() / 1e6
    ax.plot([0, spend_max * 1.1], [0, spend_max * 1.1 * 1.2], "--",
            color=COLORS["danger"], alpha=0.6, label="Break-even ROAS 1.2")
    ax.set_xlabel("Spend ($M)"); ax.set_ylabel("Revenue ($M)")
    ax.set_title("Spend vs Revenue by Platform\n(Bubble size = Purchases)", fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3)
    save_fig("channel_spend_vs_revenue")

    return summary

# ─────────────────────────────────────────────────────────────────────────────
# 2. REGIONAL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
def regional_analysis(df: pd.DataFrame):
    print("\n  [2/7] Regional performance analysis...")
    set_style()

    summary = df.groupby("Region").agg(
        Total_Spend=("Spend", "sum"),
        Revenue=("Revenue", "sum"),
        ROAS=("ROAS", "mean"),
        CTR=("CTR", "mean"),
        CVR=("CVR", "mean"),
        Purchases=("Purchases", "sum"),
    ).reset_index().sort_values("ROAS", ascending=False)

    sig = kruskal_significance(df, "Region", "ROAS")
    print(f"    Regional ROAS significance: {sig}")
    print(summary.to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    regions = summary["Region"].tolist()
    colors  = [COLORS.get(r, COLORS["primary"]) for r in regions]

    # ROAS
    ax = axes[0]
    bars = ax.bar(regions, summary["ROAS"], color=colors, edgecolor="white")
    ax.axhline(1.2, color=COLORS["danger"], linestyle="--", alpha=0.7, label="Target 1.2")
    ax.set_title(f"ROAS by Region\n({sig})", fontweight="bold")
    ax.set_ylabel("ROAS"); ax.legend(fontsize=8)
    for bar, val in zip(bars, summary["ROAS"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", fontsize=9, fontweight="bold")

    # Spend share
    ax = axes[1]
    wedges, texts, autotexts = ax.pie(
        summary["Total_Spend"], labels=regions,
        colors=colors, autopct="%1.1f%%", startangle=90,
        textprops={"fontsize": 9}
    )
    ax.set_title("Spend Distribution", fontweight="bold")

    # CVR
    ax = axes[2]
    bars = ax.bar(regions, summary["CVR"] * 100, color=colors, edgecolor="white")
    ax.set_title("Conversion Rate by Region", fontweight="bold")
    ax.set_ylabel("CVR (%)")
    for bar, val in zip(bars, summary["CVR"] * 100):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.2f}%", ha="center", fontsize=9, fontweight="bold")

    fig.suptitle(f"Regional Performance Analysis", fontweight="bold", fontsize=13)
    save_fig("regional_performance")
    return summary

# ─────────────────────────────────────────────────────────────────────────────
# 3. CREATIVE PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
def creative_analysis(df: pd.DataFrame):
    print("\n  [3/7] Creative performance analysis...")
    set_style()

    summary = df.groupby("Creative_Type").agg(
        ROAS=("ROAS", "mean"),
        CTR=("CTR", "mean"),
        CVR=("CVR", "mean"),
        CPC=("CPC", "mean"),
        Total_Spend=("Spend", "sum"),
        Purchases=("Purchases", "sum"),
    ).reset_index().sort_values("ROAS", ascending=False)

    sig = kruskal_significance(df, "Creative_Type", "ROAS")
    print(f"    Creative ROAS significance: {sig}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    types  = summary["Creative_Type"].tolist()
    colors = sns.color_palette("viridis", len(types))

    for ax, (metric, label) in zip(axes, [("ROAS","ROAS"),("CTR","CTR"),("CVR","CVR"),("CPC","CPC ($)")]):
        vals = summary[metric] * (100 if metric in ["CTR","CVR"] else 1)
        bars = ax.bar(types, vals, color=colors, edgecolor="white")
        if metric == "ROAS":
            ax.axhline(1.2, color=COLORS["danger"], linestyle="--", alpha=0.7, label="Target 1.2")
            ax.legend(fontsize=8)
        ax.set_title(f"{label} by Creative Type", fontweight="bold", fontsize=10)
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=20)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01,
                    f"{val:.2f}", ha="center", fontsize=8)

    fig.suptitle(f"Creative Type Performance  ({sig})", fontweight="bold", fontsize=13)
    save_fig("creative_performance")

    # Video completion rate vs CVR scatter
    video_df = df[df["Creative_Type"].str.lower() == "video"].dropna(
        subset=["Video_Completion_Rate", "CVR"])
    if len(video_df) > 10:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(video_df["Video_Completion_Rate"] * 100, video_df["CVR"] * 100,
                   alpha=0.4, color=COLORS["primary"], s=30)
        m, b, r, p, _ = stats.linregress(
            video_df["Video_Completion_Rate"], video_df["CVR"])
        x_range = np.linspace(0, 1, 100)
        ax.plot(x_range * 100, (m * x_range + b) * 100,
                color=COLORS["danger"], linewidth=2,
                label=f"r={r:.2f}, {mannwhitney(video_df['Video_Completion_Rate'].values[:len(video_df)//2], video_df['Video_Completion_Rate'].values[len(video_df)//2:])}")
        ax.set_xlabel("Video Completion Rate (%)"); ax.set_ylabel("CVR (%)")
        ax.set_title("Video Completion Rate vs Conversion Rate", fontweight="bold")
        ax.legend()
        save_fig("video_completion_vs_cvr")

    return summary

# ─────────────────────────────────────────────────────────────────────────────
# 4. PRODUCT & AUDIENCE
# ─────────────────────────────────────────────────────────────────────────────
def product_audience_analysis(df: pd.DataFrame):
    print("\n  [4/7] Product & audience analysis...")
    set_style()

    prod = df.groupby("Product_Category").agg(
        ROAS=("ROAS","mean"), Revenue=("Revenue","sum"),
        Total_Spend=("Spend","sum"), Purchases=("Purchases","sum"),
        LTV=("Customer_LTV","mean")
    ).reset_index().sort_values("ROAS", ascending=False)

    aud = df.groupby("Target_Audience").agg(
        ROAS=("ROAS","mean"), Revenue=("Revenue","sum"),
        CVR=("CVR","mean"), CPC=("CPC","mean")
    ).reset_index().sort_values("ROAS", ascending=False)

    # Heatmap: product × platform ROAS
    heatmap_data = df.pivot_table(values="ROAS", index="Product_Category",
                                  columns="Platform", aggfunc="mean")

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig)

    # Product ROAS bar
    ax1 = fig.add_subplot(gs[0, :2])
    colors = [roas_color(v) for v in prod["ROAS"]]
    bars = ax1.bar(prod["Product_Category"], prod["ROAS"], color=colors, edgecolor="white")
    ax1.axhline(1.2, color=COLORS["danger"], linestyle="--", alpha=0.7, label="Target 1.2")
    ax1.set_title("ROAS by Product Category", fontweight="bold")
    ax1.set_ylabel("ROAS"); ax1.legend(fontsize=8)
    for bar, val in zip(bars, prod["ROAS"]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.2f}", ha="center", fontsize=9, fontweight="bold")

    # LTV by product
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.barh(prod["Product_Category"], prod["LTV"],
             color=COLORS["primary"], edgecolor="white")
    ax2.set_title("Avg Customer LTV", fontweight="bold")
    ax2.set_xlabel("LTV ($)")

    # Audience ROAS
    ax3 = fig.add_subplot(gs[1, :2])
    colors_a = [roas_color(v) for v in aud["ROAS"]]
    bars = ax3.bar(aud["Target_Audience"], aud["ROAS"], color=colors_a, edgecolor="white")
    ax3.axhline(1.2, color=COLORS["danger"], linestyle="--", alpha=0.7)
    ax3.set_title("ROAS by Target Audience", fontweight="bold")
    ax3.set_ylabel("ROAS")
    for bar, val in zip(bars, aud["ROAS"]):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.2f}", ha="center", fontsize=9, fontweight="bold")

    # Heatmap
    ax4 = fig.add_subplot(gs[1, 2])
    sns.heatmap(heatmap_data, ax=ax4, annot=True, fmt=".2f",
                cmap="RdYlGn", center=1.2, linewidths=0.5,
                cbar_kws={"label": "ROAS"})
    ax4.set_title("ROAS Heatmap\nProduct × Platform", fontweight="bold", fontsize=9)

    fig.suptitle("Product Category & Audience Targeting Performance", fontweight="bold", fontsize=13)
    save_fig("product_audience_performance")
    return prod, aud

# ─────────────────────────────────────────────────────────────────────────────
# 5. WEEK-OVER-WEEK TRENDS
# ─────────────────────────────────────────────────────────────────────────────
def wow_trends(df: pd.DataFrame):
    print("\n  [5/7] Week-over-week trend analysis...")
    set_style()

    weekly = df.groupby(["Week", "Platform"]).agg(
        ROAS=("ROAS","mean"), Total_Spend=("Spend","sum"),
        Revenue=("Revenue","sum"), CTR=("CTR","mean")
    ).reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # ROAS by platform weekly
    ax = axes[0, 0]
    for plat, grp in weekly.groupby("Platform"):
        ax.plot(grp["Week"], grp["ROAS"], marker="o", label=plat,
                color=COLORS.get(plat), linewidth=2)
    ax.axhline(1.2, color=COLORS["danger"], linestyle="--", alpha=0.7, label="Target 1.2")
    ax.set_title("Weekly ROAS by Platform", fontweight="bold")
    ax.set_xlabel("Week"); ax.set_ylabel("ROAS"); ax.legend(fontsize=8)

    # Spend trend
    ax = axes[0, 1]
    spend_weekly = df.groupby("Week")["Spend"].sum()
    rev_weekly   = df.groupby("Week")["Revenue"].sum()
    ax.bar(spend_weekly.index, spend_weekly.values, alpha=0.6,
           color=COLORS["danger"], label="Spend")
    ax.bar(rev_weekly.index, rev_weekly.values, alpha=0.6,
           color=COLORS["ok"], label="Revenue", bottom=0)
    ax.set_title("Weekly Spend vs Revenue", fontweight="bold")
    ax.set_xlabel("Week"); ax.yaxis.set_major_formatter(plt.FuncFormatter(fmt_currency))
    ax.legend(fontsize=8)

    # CTR trend
    ax = axes[1, 0]
    for plat, grp in weekly.groupby("Platform"):
        ax.plot(grp["Week"], grp["CTR"] * 100, marker="s", label=plat,
                color=COLORS.get(plat), linewidth=2, linestyle="--")
    ax.set_title("Weekly CTR by Platform (%)", fontweight="bold")
    ax.set_xlabel("Week"); ax.set_ylabel("CTR %"); ax.legend(fontsize=8)

    # WoW ROAS change
    ax = axes[1, 1]
    overall_weekly = df.groupby("Week")["ROAS"].mean()
    wow_change = overall_weekly.pct_change() * 100
    colors_wow = [COLORS["ok"] if v >= 0 else COLORS["danger"] for v in wow_change.fillna(0)]
    ax.bar(wow_change.index, wow_change.values, color=colors_wow, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Week-over-Week ROAS Change (%)", fontweight="bold")
    ax.set_xlabel("Week"); ax.set_ylabel("Change %")

    fig.suptitle("Week-over-Week Performance Trends", fontweight="bold", fontsize=13)
    save_fig("wow_trends")

# ─────────────────────────────────────────────────────────────────────────────
# 6. COMPETITIVE EVENTS
# ─────────────────────────────────────────────────────────────────────────────
def competitive_events_analysis(df: pd.DataFrame):
    print("\n  [6/7] Competitive events impact analysis...")
    set_style()

    comp  = df[df["Is_Competitive_Event"] == True]
    ncomp = df[df["Is_Competitive_Event"] == False]

    metrics = ["ROAS", "CPC", "CPM", "CVR", "CTR"]
    results = []
    for m in metrics:
        a = comp[m].dropna()
        b = ncomp[m].dropna()
        sig = mannwhitney(a.values, b.values)
        results.append({"Metric": m, "Competitive": a.mean(), "Normal": b.mean(),
                        "Diff%": (a.mean() - b.mean()) / b.mean() * 100 if b.mean() != 0 else 0,
                        "Significance": sig})

    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Bar comparison
    ax = axes[0]
    x = np.arange(len(metrics))
    w = 0.35
    comp_vals  = [comp[m].mean() for m in metrics]
    ncomp_vals = [ncomp[m].mean() for m in metrics]
    ax.bar(x - w/2, comp_vals,  w, label="Competitive Period",  color=COLORS["danger"],  alpha=0.8)
    ax.bar(x + w/2, ncomp_vals, w, label="Normal Period",       color=COLORS["primary"], alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_title("Performance: Competitive vs Normal Periods", fontweight="bold")
    ax.legend(fontsize=8)

    # CPM over time with competitive overlay
    ax = axes[1]
    weekly_cpm  = df.groupby("Week")["CPM"].mean()
    comp_weeks  = df[df["Is_Competitive_Event"] == True]["Week"].unique()
    ax.plot(weekly_cpm.index, weekly_cpm.values, color=COLORS["primary"], linewidth=2, marker="o")
    for w_idx in comp_weeks:
        ax.axvspan(w_idx - 0.5, w_idx + 0.5, alpha=0.15, color=COLORS["danger"])
    ax.set_title("CPM Over Time\n(shaded = competitive events)", fontweight="bold")
    ax.set_xlabel("Week"); ax.set_ylabel("CPM ($)")

    fig.suptitle("Impact of Competitive Events on Performance", fontweight="bold", fontsize=13)
    save_fig("competitive_events")
    return res_df

# ─────────────────────────────────────────────────────────────────────────────
# 7. FREQUENCY vs CONVERSION
# ─────────────────────────────────────────────────────────────────────────────
def frequency_analysis(df: pd.DataFrame):
    print("\n  [7/7] Frequency vs conversion analysis...")
    set_style()

    freq_df = df.dropna(subset=["Frequency", "CVR"])
    freq_bins = pd.cut(freq_df["Frequency"], bins=[0, 1, 2, 3, 5, 10, 99],
                       labels=["0-1", "1-2", "2-3", "3-5", "5-10", "10+"])
    freq_grouped = freq_df.groupby(freq_bins, observed=True).agg(
        CVR=("CVR", "mean"), ROAS=("ROAS", "mean"),
        CPC=("CPC", "mean"), n=("CVR", "count")
    ).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    for ax, metric in zip(axes, ["CVR", "ROAS", "CPC"]):
        vals = freq_grouped[metric] * (100 if metric == "CVR" else 1)
        colors = [COLORS["ok"] if metric != "CPC" and v > vals.mean()
                  else COLORS["danger"] if metric == "CPC" and v > vals.mean()
                  else COLORS["primary"] for v in vals]
        ax.bar(freq_grouped["Frequency"].astype(str), vals, color=colors, edgecolor="white")
        ax.set_title(f"{metric} by Ad Frequency Bucket", fontweight="bold")
        ax.set_xlabel("Frequency"); ax.set_ylabel(metric + ("%" if metric == "CVR" else ""))

    # Scatter
    r, p = stats.pearsonr(freq_df["Frequency"].values, freq_df["CVR"].values)
    axes[0].set_title(f"CVR by Frequency  (r={r:.2f}, p={p:.3f})", fontweight="bold")

    fig.suptitle("Ad Frequency vs Conversion & Efficiency", fontweight="bold", fontsize=13)
    save_fig("frequency_vs_cvr")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def run():
    print(f"\n{'='*60}")
    print(f"  TASK 2 — PERFORMANCE ANALYSIS")
    print(f"{'='*60}")

    try:
        df = pd.read_csv(REPORT_DIR / "cleaned_data.csv", parse_dates=["Date"])
    except FileNotFoundError:
        print("  cleaned_data.csv not found — running Task 1 first...")
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location("dq", Path(__file__).parent / "01_data_quality.py")
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        df   = mod.run()

    channel_summary  = channel_analysis(df)
    regional_summary = regional_analysis(df)
    creative_summary = creative_analysis(df)
    prod_summary, aud_summary = product_audience_analysis(df)
    wow_trends(df)
    comp_results = competitive_events_analysis(df)
    frequency_analysis(df)

    # Combined performance summary
    all_summaries = {
        "channel":  channel_summary,
        "region":   regional_summary,
        "creative": creative_summary,
        "product":  prod_summary,
        "audience": aud_summary,
    }
    with pd.ExcelWriter(REPORT_DIR / "performance_summary.xlsx") as writer:
        for sheet, summ in all_summaries.items():
            summ.to_excel(writer, sheet_name=sheet.capitalize(), index=False)
    print(f"\n  [saved] performance_summary.xlsx")
    print(f"\n  ✓ Task 2 complete.\n")
    return df, all_summaries

if __name__ == "__main__":
    run()
