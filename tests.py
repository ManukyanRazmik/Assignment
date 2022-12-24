"""
Unittests to test Trainig dataset, Test dataset and Ideal functions
"""

import unittest
from Datasets import *
from pandas._testing import assert_frame_equal

TRAIN_PATH = 'data/train.csv'
IDEAL_FUNC_PATH = 'data/ideal.csv'
TEST_PATH = 'data/test.csv'

COLS = ['y1', 'y2', 'y3', 'y4']
TESTDF = pd.DataFrame(np.arange(-20.0, 20.0, 0.1), columns=['x'])

class TestIdeal(unittest.TestCase):

    def setUp(self):
        self.ideal = IdealFunctionData(IDEAL_FUNC_PATH)
        
    def test_x(self):        
        assert_frame_equal(self.ideal.x, TESTDF, 'Values of x are different')        
        
        

class TestTrain(unittest.TestCase):

    def setUp(self):
        self.train = TrainData(TRAIN_PATH, 'y2' )
        
    def test_col_names(self):                
        self.assertEqual(sorted(self.train._func_names), COLS, 'Function names are different')
        
    def test_fit(self):
        self.ideal = IdealFunctionData(IDEAL_FUNC_PATH)
        self.train.fit_ideal(self.ideal)
        self.assertEqual(self.train.max_dev.round(2), 0.5, 'Data was fitted incorrectly')
        self.assertEqual(self.train.ideal_y.name, 'y42', 'Data was fitted incorrectly')
        

        
class TestTest(unittest.TestCase):

    def setUp(self):
        self.test = TestData(TEST_PATH)
        
    def test_df(self):
        self.assertEqual(list(self.test._df.columns), ['x', 'y'], 'Different columns provided')
        
    def test_fit(self):
        self.ideal = IdealFunctionData(IDEAL_FUNC_PATH)
        self.train = TrainData(TRAIN_PATH, 'y2' )
        self.train.fit_ideal(self.ideal)
        self.test.fit(self.ideal, self.train)
        self.assertEqual(self.test.testDB.iloc[4, 2][0].round(2), 0.43, 'Fitting was incorrect')
        self.assertEqual(len(self.test.testDB),len(self.test._df), 'Fitting was incorrect')        
    
if __name__ == '__main__':
    unittest.main()