"""Training-only exploration of repayment history and default."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from .uci_data import load_data, clean_data, split_data, PAY_STATUS, TARGET


def main():
    raw=load_data('data/uci/credit_card_clients.csv')
    cleaned=clean_data(raw)
    train,_,_=split_data(cleaned)
    out=Path('reports/uci_eda');out.mkdir(parents=True,exist_ok=True)
    (out/'quality.json').write_text(json.dumps(dict(raw_rows=len(raw),deduplicated_rows=len(cleaned),training_rows=len(train),raw_default_rate=float(raw[TARGET].mean()),missing=raw.isna().sum().to_dict()),indent=2))
    train.describe(include='all').to_csv(out/'training_summary.csv')
    sns.set_theme(style='whitegrid')
    def save(name):
        plt.tight_layout();plt.savefig(out/f'{name}.png',dpi=140);plt.close()
    sns.countplot(data=train,x=TARGET);plt.title('Training labels: 1 = next-month default');save('class_balance')
    for col in ['LIMIT_BAL','AGE']:
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        sns.histplot(data=train,x=col,hue=TARGET,stat='density',common_norm=False,ax=axes[0])
        sns.boxplot(data=train,x=TARGET,y=col,showfliers=False,ax=axes[1]);save(col.lower())
    for col in ['PAY_0','EDUCATION','MARRIAGE']:
        t=train.groupby(col)[TARGET].agg(count='size',default_rate='mean');t.to_csv(out/f'{col}_rates.csv')
        fig,axes=plt.subplots(1,2,figsize=(10,4));t['count'].plot.bar(ax=axes[0],title='Group count');t.default_rate.plot.bar(ax=axes[1],title='Observed default rate');save(f'{col}_rates')
    history=train[PAY_STATUS].astype(int).gt(0).sum(axis=1)
    t=train.groupby(history)[TARGET].agg(count='size',default_rate='mean');t.to_csv(out/'delayed_months.csv')
    t.default_rate.plot.bar();plt.xlabel('Months with positive delay code');plt.ylabel('Observed default rate');save('delayed_months')
    for prefix in ['BILL_AMT','PAY_AMT']:
        cols=[f'{prefix}{i}' for i in range(6,0,-1)]
        train.groupby(TARGET)[cols].mean().T.plot(marker='o',figsize=(9,4));plt.xlabel('April → September 2005');plt.ylabel('Mean amount (NT dollars)');save(prefix.lower()+'_history')
    (out/'README.md').write_text('Plots use only the grouped training split. Group rates are associations, not causal effects. Read rates together with group counts. Special category codes are preserved without assigning undocumented meanings. ID and target are excluded from all predictors. Delayed-month analysis uses positive repayment status codes; these engineered features are not used in the reported baseline models.\n')
    print(f'EDA saved to {out}')

if __name__=='__main__': main()
