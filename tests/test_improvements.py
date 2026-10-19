import json
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest
from src.data_cleaning import ROOT, RAW, CLEAN, clean_laptops
from src.pricing_engine import compare_laptop
from src.prediction import validate_specs
from src.train_model import FEATURES, model_groups

class Improvements(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d=pd.read_csv(CLEAN)
        cls.reference=pd.read_csv(ROOT/'reports/training_reference.csv')

    def test_original_files_preserved(self):
        import hashlib
        manifest=json.loads((ROOT/'reports/reproducibility.json').read_text())
        self.assertEqual(hashlib.sha256(RAW.read_bytes()).hexdigest(),manifest['raw_sha256'])
        self.assertTrue((ROOT/'original_lap.py').is_file())
        self.assertTrue((ROOT/'evaluation/original_model_metrics.json').is_file())

    def test_group_split_is_disjoint(self):
        split=pd.read_csv(ROOT/'reports/split_manifest.csv')
        self.assertFalse(set(split[split.split=='train'].group)&set(split[split.split=='test'].group))
        self.assertEqual(len(split),len(self.d))

    def test_strict_comparison_and_sparse_peers(self):
        row=self.d.iloc[[0]].copy()
        small=pd.concat([row]*4,ignore_index=True)
        small.loc[1:,'Price']=row.Price.iloc[0]*2
        r=compare_laptop(small,0)
        self.assertEqual(r['peer_count'],3)
        self.assertEqual(r['relative_pct'],-50)
        small.loc[1,'CPU']='Different CPU'
        r=compare_laptop(small,0)
        self.assertEqual(r['peer_count'],2)
        self.assertIsNone(r['peer_median'])
        small.index=[10,20,30,40]
        self.assertEqual(compare_laptop(small,10)['peer_count'],2)
        with self.assertRaises(ValueError): compare_laptop(small,0)

    def test_manual_validation(self):
        specs=self.reference.iloc[0].to_dict()
        frame,warnings=validate_specs(specs,self.reference)
        self.assertEqual(list(frame.columns),FEATURES)
        for value in [-1,float('inf'),float('nan')]:
            with self.assertRaises(ValueError): validate_specs(dict(specs,RAM_GB=value),self.reference)
        with self.assertRaises(ValueError): validate_specs(dict(specs,Total_Storage_GB=1),self.reference)
        with self.assertRaises(ValueError): validate_specs(dict(specs,Manufacturer='unseen'),self.reference)
        _,warnings=validate_specs(dict(specs,RAM_GB=10000),self.reference)
        self.assertTrue(warnings)

    def test_cleaning_invalid_schema_and_prices(self):
        raw=pd.read_csv(RAW)
        with self.assertRaises(ValueError): clean_laptops(raw.drop(columns='Operating System'))
        raw.loc[0,'Price']=np.inf
        self.assertTrue(np.isfinite(clean_laptops(raw).Price).all())

    def test_streamlit_workflows(self):
        app=AppTest.from_file(str(ROOT/'app.py')).run(timeout=30)
        self.assertFalse(list(app.exception))
        self.assertGreaterEqual(len(app.metric),10)
        app.button[0].click().run(timeout=30)
        self.assertFalse(list(app.exception))
        self.assertTrue(any('Estimated price' in item.label for item in app.metric))
        app.number_input[0].set_value(0)
        app.button[0].click().run(timeout=30)
        self.assertTrue(list(app.error))
        self.assertFalse(list(app.exception))
        # Empty filter intersection should display a useful warning.
        brand=self.d.Manufacturer.iloc[0]
        unavailable=next(((b,c) for b in self.d.Manufacturer.unique() for c in self.d.Category.unique() if self.d[(self.d.Manufacturer==b)&(self.d.Category==c)].empty),None)
        if unavailable:
            app.multiselect[0].set_value([unavailable[0]])
            app.multiselect[1].set_value([unavailable[1]])
            app.run(timeout=30)
            self.assertTrue(any('No records' in warning.value for warning in app.warning))
        self.assertFalse(list(app.exception))

    def test_reported_holdout_metrics(self):
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        report=json.loads((ROOT/'reports/model_metrics.json').read_text())
        scores=report[report['selected_model']]
        values=pd.read_csv(ROOT/'reports/holdout_predictions.csv')
        self.assertAlmostEqual(scores['test_mae'],mean_absolute_error(values.actual,values.predicted))
        self.assertAlmostEqual(scores['test_rmse'],np.sqrt(mean_squared_error(values.actual,values.predicted)))
        self.assertAlmostEqual(scores['test_r2'],r2_score(values.actual,values.predicted))

    def test_selection_receives_training_rows_only(self):
        from src.train_model import evaluate
        split=pd.read_csv(ROOT/'reports/split_manifest.csv')
        train=split[split.split=='train'].row_id.to_numpy()
        test=split[split.split=='test'].row_id.to_numpy()
        def cv_check(pipe,x,y,groups,scoring,cv,n_jobs):
            self.assertEqual(set(x.index),set(train))
            self.assertFalse(set(x.index)&set(test))
            for left,right in cv.split(x,y,groups):
                self.assertFalse(set(groups.iloc[left])&set(groups.iloc[right]))
            return -np.ones(5)
        with patch('src.train_model.cross_val_score',side_effect=cv_check) as scoring:
            result,_,_=evaluate(self.d,train,test,grouped=True)
        self.assertEqual(scoring.call_count,3)
        self.assertEqual(result['selected_model'],'Baseline median')

    def test_missing_model_message(self):
        with patch('joblib.load',side_effect=FileNotFoundError('test missing model')):
            app=AppTest.from_file(str(ROOT/'app.py')).run(timeout=30)
        self.assertFalse(list(app.exception))
        self.assertTrue(any('Model artifacts unavailable' in item.value for item in app.warning))

    def test_missing_dataset_message(self):
        original=pd.read_csv
        def read(path,*args,**kwargs):
            if PathLike(path)==str(CLEAN): raise FileNotFoundError('test missing data')
            return original(path,*args,**kwargs)
        with patch('pandas.read_csv',side_effect=read):
            app=AppTest.from_file(str(ROOT/'app.py')).run(timeout=30)
        self.assertFalse(list(app.exception))
        self.assertTrue(list(app.error))


def PathLike(value):
    return str(value)
