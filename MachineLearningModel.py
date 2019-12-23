import numpy as np

class MachineLearningModel:
  '''
  MachineLearningModel is a superclass to all other Models.

  Args:
    X (np.array of shape (n, m))  : n rows of data points down x m rows of feature columns
    Y (np.array of shape (n, ))   : n rows of y target values
  '''
  def __init__(self, X, Y):
    self.X = X
    self.Y = Y
    self.n = X.shape[0]

    print("Initializing MachineLearningModel...")
    print(f"X shape: {self.X.shape}, Y shape: {self.Y.shape}")
    print(f"X: \n {self.X}")
    print(f"Y: \n {self.Y}")

if __name__ == "__main__":
  # get data as np.array
  x = np.array([[0.2, 0.3],
                [0.1, 0.7]])
  y = np.array([0.2, 0.1])

  # help(MachineLearningModel)

  model = MachineLearningModel(x, y)
