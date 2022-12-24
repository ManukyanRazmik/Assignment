# Importing necessary modules an functions

import pandas as pd
import numpy as np
import sqlalchemy as db
from sqlalchemy_utils import database_exists, create_database
from sqlalchemy import create_engine
from bokeh.plotting import figure, output_file, show, gridplot, reset_output
from bokeh.io import export_png, curdoc
from Datasets import DataHandler, TrainData, IdealFunctionData, TestData
from utils import *

# Defining constants

# Paths to datasets
TRAIN_PATH = 'data/train.csv'
IDEAL_FUNC_PATH = 'data/ideal.csv'
TEST_PATH = 'data/test.csv'

# Database credentials
PROTOCOL = 'postgresql+psycopg2'
USER = 'razmik'
PASS = 'razmik_010101'
HOST = 'localhost'
DB = 'assignment'


# Creating engine
eng = dbEngine(PROTOCOL, USER, PASS, HOST, DB)

# Defining objects of Traindata and Ideal objects

train = DataHandler(TRAIN_PATH)
ideal = IdealFunctionData(IDEAL_FUNC_PATH)

# Create table and load the training data and ideal functions

train.toDB(eng, 'train')
ideal.toDB(eng, 'ideal')

# Create 4 different objects for each training dataset

x1 = TrainData(TRAIN_PATH, 'y1')
x2 = TrainData(TRAIN_PATH, 'y2')
x3 = TrainData(TRAIN_PATH, 'y3')
x4 = TrainData(TRAIN_PATH, 'y4')

# Fitting ideal functions to each trainig data 

x1.fit_ideal(ideal)
x2.fit_ideal(ideal)
x3.fit_ideal(ideal)
x4.fit_ideal(ideal)

# Plotting Training data and group of Ideal functions

x1.plot_line(path = r'results/train')
x2.plot_line(path = r'results/train')
x3.plot_line(path = r'results/train')
x4.plot_line(path = r'results/train')

ideal.plot_all(path = r'results/ideal')


# Plotting training data with fitted to it ideal function

x1.train_ideal_plot(path = r'results/train')
x2.train_ideal_plot(path = r'results/train')
x3.train_ideal_plot(path = r'results/train')
x4.train_ideal_plot(path = r'results/train')


# Create object of Test dataset

test = TestData(TEST_PATH)

# Fit best functions to each data in test dataset

test.fit(ideal, x1, x2, x3, x4)

# Look at the resulting dataframe

test.testDB

# Load the resulting dataframe into database

test.toDB(eng, 'test')

# And, finally, plot of test points

test.plot_points()