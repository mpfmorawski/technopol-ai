import pandas as pd


class Preprocessing(object):
    """
    Class for first preprocessing of given data
    """

    def __init__(self, path):
        self.df = self.read_data(path)

    def read_data(self, path):
        return pd.read_csv(path, on_bad_lines = "skip")

    def trim_redundant_data(self):
        columns_to_drop = [
            'BBLE',
            'OWNER',
            'PERIOD',
            'Borough',
            'New Georeferenced Column'
        ]
        return self.df.drop(columns = columns_to_drop)


    def run(self):
        self.df = self.trim_redundant_data()
        return self.df
