# DTC Marketing Analytics — Automated Assessment Pipeline

## Project Structure

```
marketing_analysis/
├── data/
│   └── marketing_data.xlsx          ← Place your Excel dataset here
├── src/
│   01_data_quality.py               ← Task 1: Cleaning & EDA
│   02_performance_analysis.py       ← Task 2: Statistical Analysis & Charts
│   03_recommendations.py            ← Task 3: Strategic Recommendations Doc
│   04_predictive_model.py           ← Task 4: Bonus Predictive Model
│   utils.py                         ← Shared helpers
├── outputs/
│   ├── figures/                     ← All charts saved here
│   └── reports/                     ← Word docs, CSVs saved here
├── run_all.py                       ← Single entry point: runs everything
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Run Everything

```bash
python run_all.py
```

## Run Individual Tasks

```bash
python src/01_data_quality.py
python src/02_performance_analysis.py
python src/03_recommendations.py
python src/04_predictive_model.py
```

## Outputs

| File                                        | Description                              |
| ------------------------------------------- | ---------------------------------------- |
| `outputs/reports/data_quality_report.csv` | Missing values, outliers, cleaning log   |
| `outputs/reports/performance_summary.csv` | KPIs by platform/region/creative/product |
| `outputs/figures/*.png`                   | All analysis charts                      |
| `outputs/reports/executive_summary.docx`  | Final Word report (Task 3)               |
| `outputs/reports/model_results.csv`       | ROAS predictions & feature importance    |
