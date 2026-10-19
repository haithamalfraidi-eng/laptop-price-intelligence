"""Clean and engineer laptop features, without changing the original files."""
import re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw' / 'laptops_test.csv'
CLEAN = ROOT / 'data' / 'processed' / 'laptops_clean.csv'

def storage_gb(value, kind):
    result = 0.0
    for number,unit,typ in re.findall(r'(\d+(?:\.\d+)?)\s*(TB|GB)\s*(SSD|HDD|Hybrid|Flash Storage)', str(value), flags=re.I):
        if typ.lower() == kind.lower():
            result += float(number) * (1000 if unit.upper()=='TB' else 1)
    return result

def clean_laptops(df):
    d=df.copy()
    d.columns=d.columns.str.strip()
    required=['Manufacturer','Model Name','Category','Screen Size','CPU','RAM','Storage','GPU','Weight','Price','Operating System']
    missing=[c for c in required if c not in d]
    if missing: raise ValueError(f'Missing required columns: {missing}')
    d=d.drop_duplicates().reset_index(drop=True)
    d['RAM_GB']=pd.to_numeric(d['RAM'].astype(str).str.replace('GB','',regex=False),errors='coerce')
    d['Screen_Inches']=pd.to_numeric(d['Screen Size'].astype(str).str.replace('"','',regex=False),errors='coerce')
    d['Weight_KG']=pd.to_numeric(d['Weight'].astype(str).str.replace('kg','',regex=False),errors='coerce')
    d['Price']=pd.to_numeric(d['Price'],errors='coerce')
    d['SSD_GB']=d['Storage'].map(lambda x: storage_gb(x,'SSD'))
    d['HDD_GB']=d['Storage'].map(lambda x: storage_gb(x,'HDD'))
    d['Total_Storage_GB']=d['Storage'].map(lambda s: sum(storage_gb(s,k) for k in ['SSD','HDD','Hybrid','Flash Storage']))
    d['CPU_Brand']=d['CPU'].fillna('Unknown').astype(str).str.extract(r'^(Intel|AMD|Samsung)',expand=False).fillna('Other')
    d['OS_Version']=d.get('Operating System Version',pd.Series(index=d.index,dtype='object')).fillna('Not Specified').astype(str)
    d=d.dropna(subset=['Price','RAM_GB','Screen_Inches'])
    d=d[np.isfinite(d['Price']) & np.isfinite(d['RAM_GB']) & np.isfinite(d['Screen_Inches'])]
    d=d[(d['Price']>0) & (d['RAM_GB']>0) & (d['Screen_Inches']>0)].reset_index(drop=True)
    return d

def quality_report(raw,clean):
    return {'rows_raw':len(raw),'rows_clean':len(clean),'duplicates_raw':int(raw.duplicated().sum()),
      'missing_by_column':raw.isna().sum().astype(int).to_dict(),
      'weight_unparseable_clean':int(clean['Weight_KG'].isna().sum()),
      'price_min':float(clean.Price.min()),'price_max':float(clean.Price.max()),
      'price_unit':'Unknown (source is not verified)'}

def main():
    import json
    raw=pd.read_csv(RAW)
    clean=clean_laptops(raw)
    CLEAN.parent.mkdir(parents=True,exist_ok=True)
    clean.to_csv(CLEAN,index=False)
    (ROOT/'reports').mkdir(exist_ok=True)
    (ROOT/'reports/data_quality.json').write_text(json.dumps(quality_report(raw,clean),indent=2),encoding='utf8')
    print('Saved',CLEAN,'rows:',len(clean))
if __name__=='__main__':main()
