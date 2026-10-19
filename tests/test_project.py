import unittest
import pandas as pd
from src.data_cleaning import clean_laptops,storage_gb,RAW
from src.pricing_engine import compare_laptop

class TestProject(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw=pd.read_csv(RAW)
        cls.clean=clean_laptops(cls.raw)

    def test_duplicates_removed(self):
        self.assertEqual(len(self.clean),len(self.raw.drop_duplicates()))
    def test_tb_conversion(self):
        self.assertEqual(storage_gb('2TB HDD','HDD'),2000)
        self.assertEqual(storage_gb('256GB SSD + 1TB HDD','SSD'),256)
        self.assertEqual(storage_gb('256GB SSD + 1TB HDD','HDD'),1000)
    def test_price_positive(self):
        self.assertTrue(self.clean.Price.gt(0).all())
    def test_compare(self):
        self.assertGreaterEqual(compare_laptop(self.clean,0)['peer_count'],0)

if __name__=='__main__':unittest.main()
