"""Dataset comparisons; never a quality or purchase recommendation."""
import pandas as pd

def compare_laptop(df, row_index, n=5):
    if df.empty or row_index not in df.index:
        raise ValueError('Select a valid laptop row.')
    if not isinstance(n, int) or n < 1:
        raise ValueError('Alternative count must be positive.')
    row = df.loc[row_index]
    candidates = df.drop(index=row_index).copy()
    keys = ['Category','RAM_GB','CPU','SSD_GB','HDD_GB','Total_Storage_GB']
    mask = candidates[keys].eq(row[keys]).all(axis=1) & candidates[keys].notna().all(axis=1)
    peers = candidates[mask]
    median = float(peers.Price.median()) if len(peers) >= 3 else None
    alt = candidates[candidates.Category.eq(row.Category)].copy()
    alt['Same CPU'] = alt.CPU.eq(row.CPU)
    alt['RAM difference (GB)'] = (alt.RAM_GB-row.RAM_GB).abs()
    alt['Same storage configuration'] = alt[keys[-3:]].eq(row[keys[-3:]]).all(axis=1)
    alt['Storage difference (GB)'] = (alt.Total_Storage_GB-row.Total_Storage_GB).abs()
    alt = alt.sort_values(['Same CPU','RAM difference (GB)','Same storage configuration','Storage difference (GB)'], ascending=[False,True,False,True],kind='stable')
    cols=['Manufacturer','Model Name','Category','CPU','RAM_GB','Storage','Total_Storage_GB','Price','Same CPU','Same storage configuration','RAM difference (GB)','Storage difference (GB)']
    return dict(group='Same category, RAM, exact CPU text, SSD/HDD and total capacity',peer_count=len(peers),peer_median=median,relative_pct=(float(row.Price)/median-1)*100 if median else None,alternatives=alt.head(n)[cols])
