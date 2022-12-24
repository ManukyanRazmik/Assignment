"""
Module to handel Train data, Ideal functions and Test data
"""

import pandas as pd
import numpy as np
from bokeh.plotting import figure, output_file, show, gridplot, reset_output
from bokeh.io import export_png, curdoc
from bokeh.models import Range1d
import sqlalchemy as db
from sqlalchemy_utils import database_exists, create_database
from utils import *



class DataHandler():
    """
    Base class to work with datasets

    ----------------
    path: path to dataset
    x_col: name of the column containing values of x

   
    Methods
    ----------------
    show_data: returning specified number of lines from dataset
    toDB: moving dataset to db

    """
    def __init__(self, path, x_col = 'x'):
        try:
            self._df = pd.read_csv(path)
            self.data = self._df.set_index(x_col)
            self.x = self._df[[x_col]] 
            self._func_names = list(self.data.columns)
        except FileNotFoundError as ferr:
            print(f'Directory {path} does not exist')
        except KeyError as kerr:
            print(f'Column name {x_col} is incorrect')
        except EmptyDataError as empty:
            print('The file is empty')




    def show_data(self, head = 5):
        return self.data.head(head)
    
    def toDB(self, engine, db_name):
        try:
            new_cols = list(self._df.columns[0].upper()) + [i.upper() + f' ({db_name} func) ' for i in self._df.columns[1:]]
            self._df.set_axis(new_cols, axis = 1).to_sql(db_name, engine, index = True, if_exists='replace')
        except ArgumentError as argerr:
            print('Incorrect engine')
    



class IdealFunctionData(DataHandler):
    """
    Class to create object of ideal function dataset
    -------------------
    path: path to dataset
    x_col: name of the column containing values of x


    Mehods
    -------------------
    plotline: plotting x and specified y pairs
    plot_all: plotting all ideal functions
    data_to_fit: returns x values
    """
    def plotline(self, func_name):
        try:            
            p = (self.data[func_name].min() + self.data[func_name].max()) / 2
            graph = figure(x_axis_label='x', y_axis_label='y',width=600,height=500)
            graph.x_range = Range1d(self.x['x'].min() - 3, self.x['x'].max() + 3)
            graph.y_range = Range1d(self.data[func_name].min() - p, self.data[func_name].max() + p)
            graph.line(self.x.iloc[:,0], self.data.loc[:,func_name], legend_label = f'Ideal function {func_name}', line_width = 1.5)
            return graph
        except IndexError as inderr:
            print('Name of function is incorrect')
        
    def plot_all(self, path = None, n_cols = 5, width = 500, height= 350, size=2, color='#A020F0', line_width = 0.5, line_dash=[6, 6]):
        row = []
        length = len(self.data.columns)
        for i in range(length):            
            graph = figure(title = f'Ideal Function {i + 1}', x_axis_label='x', y_axis_label='y', width = width, height= height)
            graph.align = 'center'
            graph.circle(self.x.iloc[:,0],  self.data.iloc[:,i], size=size, color= color)
            graph.line(self.x.iloc[:,0], self.data.iloc[:,i], line_width = line_width, line_dash = line_dash)
            row.append(graph)          
        gradplot = gridplot(row, ncols=n_cols)
        if path:
            output_file(f'{path}/ideals.html')
        else:
            output_file('ideals.html')
        show(gradplot)    

    def data_to_fit(self, x_index):
        return self.data.loc[x_index]      




class TrainData(DataHandler):
    """
    Class to create object of Train dataset
    -------------------
    path: path to dataset
    y_col: name of the column containing values of y
    x_col: name of the column containing values of x


    Mehods
    -------------------
    plot_line: plotting x and y pairs of train dataset
    fit_ideal: fitting one of ideal functions to trainng dataset
    train_ideal_plot: plotting training dataset and ideal finction assigned to it 
    """
    def __init__(self, path, y_col, x_col = 'x'):
        super().__init__(path, x_col = 'x')
        self.__fit = False
        try:
            if isinstance(y_col, str):            
                self.y = self.data[y_col]  
                self.y_name = y_col
            else:            
                self.y = self.data.iloc[:, y_col-1]
                self.y_name = self._func_names[y_col]
        except IndexError as inderr:
            print('Number of function is incorrect')
        
    def plot_line(self, path = None, width=600,height=500, color = '#000000', size = 4):
        number_of_func = self._func_names.index(self.y_name) + 1
        graph = figure(x_axis_label='x', y_axis_label='y',width=width,height=height)
        graph.scatter(self.x.iloc[:,0], self.y, color=color, size=size, legend_label='Training Data {}'.format(number_of_func))
        if path:
            output_file(f'{path}/training data {number_of_func}.html')
        else:
            output_file(f'training data {number_of_func}.html')
        show(graph)

    @property
    def fitted(self):        
        return self.__fit
        
    def fit_ideal(self, ideal):
        try:     
            if not isinstance(ideal, IdealFunctionData):
                raise ValueError('Not an instance of IdealFunctionData class')
            acc = np.inf
            ideals = ideal.data_to_fit(self.x.iloc[:,0])
            for col in ideal._func_names:                   
                self.sse = sum((ideals[col] - self.y) ** 2)           
                if self.sse <= acc:
                    acc = self.sse
                    self.fit_func = col
            self.max_dev = abs(ideals[self.fit_func] - self.y).max()
            self.ideal_y = ideals[self.fit_func]
            self.__fit = True
            return self
        except ValueError as verr:
            print(verr)

    def train_ideal_plot(self, path = None, width2 = 600, height2 = 500, color1 = '#000000', color2 = 'blue', size2 = 4):
        try:
            if not self.__fit:
                raise NotFittedError
            graph2 = figure(x_axis_label='x', y_axis_label='y',width=width2,height=height2)        
            graph2.scatter(self.x.iloc[:,0], self.y, color=color1, size=size2, legend_label='Training Data {}'.format(self.y_name))
            graph2.scatter(self.x.iloc[:,0], self.ideal_y, color=color2, size=size2, legend_label='Ideal Function Data {}'.format(self.fit_func), marker="x")
            if path:
                output_file(f'{path}/train_ideal plot {self.y_name}.html')
            else:
                output_file(f'train_ideal plot {self.y_name}.html')        
            show(graph2)
        except NotFittedError as fiterr:
            print('Data has not fitted yet')


        


class TestData():
    """
    Class to create object of Test dataset
    -------------------
    path: path to dataset
    y_col: name of the column containing values of y
    x_col: name of the column containing values of x


    Mehods
    -------------------

    fit: choosing mapping function(s) from 4 ideal functions found for trainig datasets
    plot_points: plotting test points and mapping function
    toDB: writing table of test dataset, mapping functions and appropriate deviations into db
    """
    def __init__(self, path, x_col = 'x', y_col = 'y'):        
        try:
            self.__fit = False
            self._df = pd.read_csv(path)
            if x_col not in self._df.columns:
                raise KeyError('{x_col} is not correct column name')
            if y_col not in self._df.columns:
                raise KeyError('{y_col} is not correct column name')
            self.x = self._df[x_col]
            self.y = self._df[y_col]
        except FileNotFoundError as ferr:
            print(f'Directory {path} does not exist')
        except KeyError as kerr:
            print(kerr)
        except EmptyDataError as empty:
            print('The file is empty')

    
    def fit(self, ideals, *fitted):        
        try:
            if not isinstance(ideals, IdealFunctionData):
                    raise ValueError('Not an instance of IdealFunctionData class')
            if not all(ids.fitted for ids in fitted):
                raise NotFittedError
            self._ideal_funcs = set()
            self._ideals = ideals
            self.testDB = pd.DataFrame([], columns = ['X (test func)', 'Y (test func)', 'Delta Y (test func)', 'No. of ideal func'], index = range(len(self.y)))
            for i, (test_x, test_y) in enumerate(zip(self.x, self.y)):           
                map_func = []
                devs = []
                for trained in fitted:                
                    func, dev = trained.fit_func, trained.max_dev
                    new_dev = abs(test_y - ideals.data.loc[test_x, func])                
                    if (new_dev <= (np.sqrt(2) * dev)):
                        devs.append(new_dev)
                        map_func.append(func)
                        self._ideal_funcs.add(func)                        
                if not devs:
                    devs = None
                    map_func = None    
                self.testDB.iloc[i] = [test_x, test_y, devs, map_func]   
            self.__fit = True
            return self
        except ValueError as verr:
            print(verr)
        except NotFittedError as fiterr:
            print('Train data has not fitted yet')
            
    def plot_points(self, path = None):
        try:            
            if not self.__fit:
                raise NotFittedError            
            plots = []
            defined = self.testDB.dropna(subset='No. of ideal func')
            for col in self._ideal_funcs:                
                mapped = defined[defined.apply(lambda x: col in x['No. of ideal func'], axis = 1)]
                not_mapped = self.testDB.drop(index = mapped.index)                
                graph = self._ideals.plotline(col)                
                graph.circle(mapped['X (test func)'], mapped['Y (test func)'], legend_label = 'Matching points', color = 'orange', size = 7, line_color = 'black', line_width = 1.5)
                graph.circle(not_mapped['X (test func)'], not_mapped['Y (test func)'], legend_label = 'Not matching points', color = 'grey')
                plots.append(graph)
            if path:
                output_file(f'{path}/test_ideals.html')
            else:
                output_file('test_ideals.html')                
            show(gridplot(plots, ncols=2))
        except NotFittedError as fiterr:
            print('Test data has not fitted yet')
        
    def toDB(self, engine, db_name):
        try:
            if not self.__fit:
                raise NotFittedError
            self.testDB.to_sql(db_name, engine, index = True, if_exists='replace')
        except ArgumentError as argerr:
            print('Incorrect engine')
        except NotFittedError as fiterr:
            print('Test data has not fitted yet')