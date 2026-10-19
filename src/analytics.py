"""EDA summaries and chart generation."""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from src.data_cleaning import ROOT, CLEAN

def summarize(df):
    return {'rows':len(df),'brands':df['Manufacturer'].nunique(),
      'median_price':float(df.Price.median()),'mean_price':float(df.Price.mean()),
      'mean_by_category':df.groupby('Category').Price.mean().sort_values(ascending=False).round(2).to_dict(),
      'median_by_ram':df.groupby('RAM_GB').Price.median().round(2).to_dict()}

def main():
    import json
    d=pd.read_csv(CLEAN)
    out=ROOT/'reports';out.mkdir(exist_ok=True)
    (out/'eda_summary.json').write_text(json.dumps(summarize(d),indent=2),encoding='utf8')
    ax=d.groupby('Category').Price.median().sort_values().plot.barh(figsize=(9,5))
    ax.set(xlabel='Median price (dataset units)',ylabel='Laptop category',title='Median laptop price by category')
    plt.tight_layout();plt.savefig(out/'category_price.png',dpi=160);plt.close()
    ax=d.plot.scatter(x='RAM_GB',y='Price',figsize=(8,5),alpha=.55)
    ax.set(xlabel='RAM (GB)',ylabel='Price (dataset units)',title='Price vs RAM')
    plt.tight_layout();plt.savefig(out/'price_vs_ram.png',dpi=160);plt.close()
    print(json.dumps(summarize(d),indent=2))
if __name__=='__main__':main()
