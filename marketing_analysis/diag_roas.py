"""Diagnose ROAS discrepancy: expected 0.86 vs computed 0.96"""
import pandas as pd, numpy as np

df = pd.read_excel("data/Marketing_Data.xlsx")

total_spend = df["Spend"].sum()
total_rev = df["Revenue"].sum()
raw_agg_roas = total_rev / total_spend

print("=== RAW DATA (no cleaning) ===")
print(f"Total Spend:   ${total_spend:,.2f}")
print(f"Total Revenue: ${total_rev:,.2f}")
print(f"Aggregate ROAS (Rev/Spend):  {raw_agg_roas:.4f}")

per_row_roas = np.where(df["Spend"] > 0, df["Revenue"] / df["Spend"], np.nan)
print(f"Mean per-row ROAS:           {np.nanmean(per_row_roas):.4f}")
print()

# Missing Revenue
n_miss_rev = df["Revenue"].isna().sum()
print(f"Missing Revenue rows: {n_miss_rev}")

# Impressions == 0
imp0 = df[df["Impressions"] == 0]
print(f"Impressions == 0 rows: {len(imp0)}")
print(f"  Of those, Revenue is non-null: {imp0['Revenue'].notna().sum()}")
print(f"  Sum of their Revenue: ${imp0['Revenue'].sum():,.2f}")
print(f"  Sum of their Spend:   ${imp0['Spend'].sum():,.2f}")
print()

# Spend outliers
q1 = df["Spend"].quantile(0.25)
q3 = df["Spend"].quantile(0.75)
iqr = q3 - q1
hi = q3 + 3 * iqr
outliers = df[df["Spend"] > hi]
print(f"Spend: Q1={q1:,.2f}, Q3={q3:,.2f}, IQR={iqr:,.2f}")
print(f"Upper fence (Q3 + 3*IQR): ${hi:,.2f}")
print(f"Spend outliers (> upper fence): {len(outliers)} rows")
print(f"  Original total Spend of outliers: ${outliers['Spend'].sum():,.2f}")
print(f"  If clipped to {hi:,.0f}: ${hi * len(outliers):,.2f}")
print(f"  Spend REDUCTION from winsorising: ${outliers['Spend'].sum() - hi * len(outliers):,.2f}")
print()

# Revenue outliers
q1r = df["Revenue"].quantile(0.25)
q3r = df["Revenue"].quantile(0.75)
iqr_r = q3r - q1r
hi_r = q3r + 3 * iqr_r
lo_r = q1r - 3 * iqr_r
out_rev = df[(df["Revenue"] > hi_r) | (df["Revenue"] < lo_r)]
print(f"Revenue: Q1={q1r:,.2f}, Q3={q3r:,.2f}, IQR={iqr_r:,.2f}")
print(f"Revenue upper fence: ${hi_r:,.2f}, lower fence: ${lo_r:,.2f}")
print(f"Revenue outliers: {len(out_rev)} rows")
print()

# Impact analysis: what happens step by step
print("=== STEP-BY-STEP ROAS IMPACT ===")
df2 = df.copy()

# Step 1: handle Impressions == 0
mask0 = df2["Impressions"] == 0
lost_rev = df2.loc[mask0, "Revenue"].sum()
df2.loc[mask0, "Clicks"] = 0
df2.loc[mask0, "Purchases"] = 0
df2.loc[mask0, "Revenue"] = np.nan
print(f"After Impressions==0 fix (set {mask0.sum()} Revenue to NaN, lost ${lost_rev:,.2f}):")
print(f"  Aggregate ROAS: {df2['Revenue'].sum() / df2['Spend'].sum():.4f}")
print()

# Step 2: Median impute Revenue (the current approach)
med_rev = df2["Revenue"].median()
n_miss = df2["Revenue"].isna().sum()
df3 = df2.copy()
df3["Revenue"] = df3["Revenue"].fillna(med_rev)
print(f"After median imputing {n_miss} Revenue rows (median=${med_rev:,.2f}):")
print(f"  Added fake revenue: ${med_rev * n_miss:,.2f}")
print(f"  Aggregate ROAS: {df3['Revenue'].sum() / df3['Spend'].sum():.4f}")
print()

# Step 3: Winsorise Spend
df4 = df3.copy()
old_spend = df4["Spend"].sum()
df4["Spend"] = df4["Spend"].clip(q1 - 3*iqr, hi)
new_spend = df4["Spend"].sum()
print(f"After winsorising Spend (clipped to [{q1-3*iqr:,.0f}, {hi:,.0f}]):")
print(f"  Spend reduced by: ${old_spend - new_spend:,.2f}")
print(f"  Aggregate ROAS: {df4['Revenue'].sum() / df4['Spend'].sum():.4f}")
print()

# Alternative: fill Revenue NaN with 0 instead of median
print("=== ALTERNATIVE: Revenue NaN -> 0 (not median) ===")
df5 = df2.copy()
df5["Revenue"] = df5["Revenue"].fillna(0)
print(f"  Aggregate ROAS (no winsorising): {df5['Revenue'].sum() / df5['Spend'].sum():.4f}")
df6 = df5.copy()
df6["Spend"] = df6["Spend"].clip(q1 - 3*iqr, hi)
print(f"  Aggregate ROAS (with winsorising): {df6['Revenue'].sum() / df6['Spend'].sum():.4f}")
print()

# Even simpler: just raw without touching Revenue at all
print("=== RAW with Revenue NaN excluded ===")
valid = df.dropna(subset=["Revenue"])
print(f"  Rows with valid Revenue: {len(valid)}")
print(f"  Aggregate ROAS: {valid['Revenue'].sum() / valid['Spend'].sum():.4f}")
