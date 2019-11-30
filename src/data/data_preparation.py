"""
Preparation of the datasets
"""

# build-ins import
import os
import numpy as np

# home-brew import
# from pandas_datareader import data


class data_set_1:
    """
    Data set consisting of handwritten_digits, cf.
    """# import source modules
    def __init__(self):
        self.name = "handwritten_digits"
        self.dir_name = os.getcwd()[:os.getcwd().] + "/src/data/raw_data/handwritten/"
        self.training_X = np.array([[]])
        self.training_Y = np.array([])
        self.test_X = np.array([[]])
        self.test_Y = np.array([])

    def process_data(self):
        for mode in ["tes", "tra"]:
            if mode == "tes":
                cross_val_set = "training"

            X_data, Y_data = [], []

            with open(self.dir_name + f"pendigits.{mode}") as raw_data:
                for line in raw_data:
                    list_line = line.replace(" ", "").replace("\n", "").split(",")
                    duets = [[int(list_line[i]), int(list_line[i+1])] for i in range(9)]
                    X_data.append(duets)
                    Y_data.append(list_line[-1])

            setattr(self, cross_val_set + "_X", np.array(X_data))
            setattr(self, cross_val_set + "_Y", np.array(Y_data))



class data_set_2:
    """
    Data set consisting of IBM stock price in 2018
    """
    def __init__(self):
        pass


# unit test
obj = data_set_1()
obj.process_data()
print(obj.training_X)