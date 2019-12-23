import os
import numpy as np
import pandas as pd

class Preprocessor:
  '''
  Preprocessor is a class containing various abstracted methods of converting raw data into the desired data structures to feed into MachineLearningModel class (e.g. X and Y data points).

  Args:
    datapath (str)  : file path to the data as a string, MUST be relative to this file!

  Attributes:
    self.datapath   : the current working directory joined with <datapath>
  '''

  def __init__(self, datapath):
    self.datapath = os.path.join(os.getcwd(), datapath)

    print("Initializing Preprocessor...")
    print(f"Loaded data from: {self.datapath}")

  def load_from_txt(self, delimiter=","):
    '''
    A method of Preprocessor
    loads data into numpy array from .txt file

    Args:
      delimiter (str)  : the delimiter as a string, default is ","

    Returns:
      self
    '''

    self.data = np.loadtxt(self.datapath, delimiter)
    return self

  def load_from_csv(self, header=None):
    '''
    A method of Preprocessor
    loads data into numpy array from .csv file

    Args:
      header (bool)  : boolean True, False or None, default is None

    Returns:
      self
    '''
    if header == None: dataframe = pd.read_csv(self.datapath)
    else: dataframe = pd.read_csv(self.datapath, header)
    self.data = np.array(dataframe)
    return self

  def get_XY(self):
    '''
    A method of Preprocessor
    splits self.data into X and Y

    Returns:
      self.X, self.Y
    '''
    self.X = self.data[:, :-1]
    self.Y = self.data[:, -1]

    return self

if __name__ == "__main__":
  datapath = os.path.join("data", "for_perceptron", "train.csv")
  data = Preprocessor(datapath).load_from_csv().get_XY()
  print(f"X: {data.X}")
  print(f"Y: {data.Y}")
