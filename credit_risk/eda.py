"""Expanded EDA: full-data quality audit, training-only statistical exploration."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from .data import load_data, clean_data, split_data, NUMERIC, CATEGORICAL, TARGET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='credit_risk_dataset.csv')
    parser.add_argument('--output', default='reports/eda')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    raw = load_data(args.data)
    cleaned = clean_data(raw)
    df, _, _ = split_data(cleaned, args.seed)
    audit = dict(raw_rows=len(raw), duplicates=int(raw.duplicated().sum()), cleaned_rows=len(cleaned),
        invalid_rows_after_dedup=len(raw.drop_duplicates())-len(cleaned), training_rows=len(df),
        missing_values=raw.isna().sum().to_dict())
    (out/'data_quality.json').write_text(json.dumps(audit, indent=2))
    df.describe(include='all').to_csv(out/'training_summary.csv')
    sns.set_theme(style='whitegrid')
    def save(name):
        plt.tight_layout()
        plt.savefig(out/f'{name}.png', dpi=140)
        plt.close()
    plt.figure(figsize=(9,4))
    raw.isna().mean().mul(100).sort_values().plot.barh()
    plt.xlabel('Missing values (%) — raw dataset')
    save('missingness')
    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x=TARGET)
    plt.title('Training labels: 0 = non-default, 1 = default')
    save('class_balance')
    for col in NUMERIC:
        fig, axes = plt.subplots(1,2,figsize=(11,4))
        sns.histplot(data=df,x=col,hue=TARGET,stat='density',common_norm=False,element='step',ax=axes[0])
        sns.boxplot(data=df,x=TARGET,y=col,ax=axes[1],showfliers=False)
        axes[1].set_title('Boxplot (outliers hidden for readability)')
        save(f'distribution_{col}')
    for col in CATEGORICAL:
        table = df.groupby(col, dropna=False)[TARGET].agg(count='size',defaults='sum',default_rate='mean').sort_values('default_rate')
        table.to_csv(out/f'rates_{col}.csv')
        fig, axes = plt.subplots(1,2,figsize=(11,4))
        table['count'].plot.bar(ax=axes[0],title='Number of applicants')
        table['default_rate'].plot.bar(ax=axes[1],title='Observed default rate')
        save(f'categories_{col}')
    plt.figure(figsize=(10,8))
    sns.heatmap(df[NUMERIC+[TARGET]].corr(),annot=True,fmt='.2f',cmap='coolwarm',center=0)
    save('numeric_correlations')
    for col in ['person_income','person_age','loan_amnt','loan_percent_income']:
        groups = pd.qcut(df[col],q=8,duplicates='drop')
        table = df.groupby(groups,observed=True)[TARGET].agg(count='size',default_rate='mean')
        table.to_csv(out/f'binned_{col}.csv')
        plt.figure(figsize=(10,4))
        table.default_rate.plot.bar()
        plt.ylabel('Observed default rate')
        save(f'binned_{col}')
    plt.figure(figsize=(10,5))
    rates=df.pivot_table(index='loan_grade',columns='loan_intent',values=TARGET,aggfunc='mean')
    sns.heatmap(rates,annot=True,fmt='.2f',cmap='YlOrRd')
    save('grade_intent_default_rate')
    df.pivot_table(index='loan_grade',columns='loan_intent',values=TARGET,aggfunc='size').to_csv(out/'grade_intent_counts.csv')
    (out/'README.md').write_text('''# EDA interpretation guide
Statistical plots use only the 60% training partition. Raw missingness and fixed-rule quality counts use the full input.
Compare default rates with sample counts; small groups are uncertain. Associations do not establish causality.
Review missing employment and interest values, extreme ages, zero income, duplicate rows and label meaning.
Loan grade and interest rate may encode earlier underwriting decisions: confirm they are available at prediction time.
The supplied dataset has no verified borrower ID or observation date: random splitting cannot establish temporal generalization or eliminate unidentified repeat-borrower leakage.
Use grade_intent_counts.csv with the heatmap. The EDA does not independently validate dataset provenance or label definitions.
''')
    print(f'EDA saved to {out}')


if __name__ == '__main__':
    main()
