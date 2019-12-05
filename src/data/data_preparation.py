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
    def __init__(self, sorting_data=True):
        self.name = "handwritten_digits"
        self.dir_name = os.getcwd().replace("/src/data/", "").replace("run_files", "src")\
                            .replace("/data", "") + "/data/raw_data/handwritten/"
        self.rescaling = 50
        self.training_X = np.array([[]])
        self.training_Y = np.array([])
        self.test_X = np.array([[]])
        self.test_Y = np.array([])

        self.training_X_ordered = {}
        self.test_X_ordered = {}

        # Data processing
        self.process_data()
        if sorting_data:
            self.data_sorting()

    def process_data(self):
        for mode in ["tes", "tra"]:
            if mode == "tes":
                cross_val_set = "training"
            elif mode == "tra":
                cross_val_set = "test"

            X_data, Y_data = [], []

            with open(self.dir_name + f"pendigits.{mode}") as raw_data:
                for line in raw_data:
                    list_line = line.replace(" ", "").replace("\n", "").split(",")
                    duets = [[int(list_line[i]) / self.rescaling, int(list_line[i+1]) / self.rescaling] for i in range(9)]
                    X_data.append(duets)
                    Y_data.append(list_line[-1])

            setattr(self, cross_val_set + "_X", np.array(X_data))
            setattr(self, cross_val_set + "_Y", np.array(Y_data))

    def data_sorting(self):
        for data_container in ["training_X", "test_X"]:
            index_tracker = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}

            for index in range(len(getattr(self, data_container))):
                number = getattr(self, data_container[:-1] + "Y")[index]
                index_tracker[int(number)].append(index)

            for i in index_tracker.keys():
                data = getattr(self, data_container)[index_tracker[i]]
                getattr(self, data_container + "_ordered")[i] = data

class data_set_2:
    """
    Data set consisting of IBM stock price in 2018
    """
    def __init__(self):
        pass

if __name__ == "__main__":
    # unit test
    obj = data_set_1()
    print(obj.training_X_ordered)
