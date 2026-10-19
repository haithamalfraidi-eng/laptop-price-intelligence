"""Load cleaned CSV into SQLite then execute reusable SQL report."""
import sqlite3
import pandas as pd
from src.data_cleaning import ROOT,CLEAN

def main():
    d=pd.read_csv(CLEAN)
    con=sqlite3.connect(ROOT/'data/processed/laptops.db')
    d.to_sql('laptops',con,if_exists='replace',index=False)
    statements=(ROOT/'sql/analysis.sql').read_text(encoding='utf8').split(';')
    with (ROOT/'reports/sql_insights.txt').open('w',encoding='utf8') as f:
        for query in statements:
            if query.strip() and not query.strip().startswith('--'):
                f.write(query.strip()+'\n'+pd.read_sql_query(query,con).to_string(index=False)+'\n\n')
    con.close()
    print('Saved reports/sql_insights.txt and data/processed/laptops.db')
if __name__=='__main__':main()
