"""Fit baseline and ML models with a held-out test set and CV on training rows only."""
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold, GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.data_cleaning import ROOT, CLEAN

NUMERIC=['RAM_GB','Screen_Inches','Weight_KG','SSD_GB','HDD_GB','Total_Storage_GB']
CATEGORICAL=['Manufacturer','Category','CPU_Brand','Operating System']
FEATURES=NUMERIC+CATEGORICAL

def make_pipeline(estimator):
    prep=ColumnTransformer([
      ('numeric',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),NUMERIC),
      ('categorical',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),
      ('onehot',OneHotEncoder(handle_unknown='ignore'))]),CATEGORICAL)])
    return Pipeline([('preprocessor',prep),('model',estimator)])

def candidates():
    return {'Baseline median':make_pipeline(DummyRegressor(strategy='median')),
            'Ridge Regression':make_pipeline(Ridge(alpha=10)),
            'Random Forest':make_pipeline(RandomForestRegressor(n_estimators=120,min_samples_leaf=3,random_state=42,n_jobs=-1))}


def evaluate(d, train, test, grouped=False):
    groups = model_groups(d)
    cv = GroupKFold(n_splits=5) if grouped else KFold(n_splits=5,shuffle=True,random_state=42)
    models = candidates()
    result = {}
    for name, pipe in models.items():
        scores = -cross_val_score(pipe,d.iloc[train][FEATURES],d.iloc[train].Price,
                                 groups=groups.iloc[train] if grouped else None,
                                 scoring='neg_mean_absolute_error',cv=cv,n_jobs=1)
        result[name] = {'cv_mae_mean':float(scores.mean()),'cv_mae_std':float(scores.std())}
    name = min(result,key=lambda key:result[key]['cv_mae_mean'])
    winner = models[name].fit(d.iloc[train][FEATURES],d.iloc[train].Price)
    actual = d.iloc[test].Price.to_numpy()
    pred = winner.predict(d.iloc[test][FEATURES])
    result[name].update(test_mae=float(mean_absolute_error(actual,pred)),
                        test_rmse=float(np.sqrt(mean_squared_error(actual,pred))),
                        test_r2=float(r2_score(actual,pred)))
    result.update(selected_model=name,train_rows=len(train),test_rows=len(test),
                  overlapping_model_groups=len(set(groups.iloc[train]) & set(groups.iloc[test])))
    return result,winner,pd.DataFrame({'row_id':test,'actual':actual,'predicted':pred})


def model_groups(d):
    return d.Manufacturer.str.strip().str.lower()+'|'+d['Model Name'].str.strip().str.lower()


def main():
    d = pd.read_csv(CLEAN)
    if len(d) < 30 or not np.isfinite(d.Price).all() or not d.Price.gt(0).all():
        raise ValueError('Training requires at least 30 valid rows with finite positive prices.')
    out = ROOT/'reports'
    out.mkdir(exist_ok=True)
    # Reproduce the original procedure exactly for an apples-to-apples comparison.
    train,test = train_test_split(np.arange(len(d)),test_size=.2,random_state=42)
    comparable,_,_ = evaluate(d,train,test)
    original = json.loads((ROOT/'evaluation/original_model_metrics.json').read_text(encoding='utf8'))
    comparison = {'original_recorded':original,'updated_same_row_split':comparable,
                  'note':'Same features, candidates, seed, row split and training-only CV. Grouped evaluation uses different rows and is not a direct improvement comparison.'}
    (out/'performance_comparison.json').write_text(json.dumps(comparison,indent=2),encoding='utf8')
    groups = model_groups(d)
    train,test = next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(d,groups=groups))
    result,winner,predictions = evaluate(d,train,test,grouped=True)
    result['evaluation'] = 'Manufacturer/model grouped holdout and training-only 5-fold GroupKFold'
    result['notes'] = 'Unknown currency/date. Exact names may not capture all related families. Small sample; no guarantee of generalization.'
    (ROOT/'models').mkdir(exist_ok=True)
    joblib.dump(winner,ROOT/'models/laptop_model.joblib')
    (out/'model_metrics.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    predictions.to_csv(out/'holdout_predictions.csv',index=False)
    d.iloc[train][FEATURES].to_csv(out/'training_reference.csv',index=False)
    pd.DataFrame({'row_id':np.arange(len(d)),'group':groups,'split':['test' if i in set(test) else 'train' for i in range(len(d))]}).to_csv(out/'split_manifest.csv',index=False)
    import hashlib, platform, importlib.metadata
    manifest = {'python':platform.python_version(),'raw_sha256':hashlib.sha256((ROOT/'data/raw/laptops_test.csv').read_bytes()).hexdigest(),
                'seed':42,'packages':{pkg:importlib.metadata.version(pkg) for pkg in ['pandas','numpy','scikit-learn','joblib','streamlit','plotly','matplotlib']}}
    (out/'reproducibility.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
