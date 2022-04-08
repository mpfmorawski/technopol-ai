import pandas as pd
import numpy as np


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
        self.replace_lengths_with_areas()
        return self.df

    def get_mean_column_value_for_boro(self, data, column_name: str, boro: int, include_zeros: bool = False):
        column_vals = data[data['BORO'] == boro][column_name]
        if include_zeros:
            return np.mean(column_vals)
        else:
            return np.mean(column_vals[column_vals != 0])

    def fill_missing_column_values_with_mean_for_boro(self, data, column_name: str):
        mean_area_per_boro = {}
        for boro in data['BORO'].unique():
            mean_area_per_boro[boro] = self.get_mean_column_value_for_boro(data, column_name, boro)

        for i, row in data.iterrows():
            data.at[i, column_name] = mean_area_per_boro[row['BORO']]

    def replace_lengths_with_areas(self):
        lot_f = 'LTFRONT'
        lot_d = 'LTDEPTH'
        bld_f = 'BLDFRONT'
        bld_d = 'BLDDEPTH'
        self.df['LTAREA'] = self.df[lot_f] * self.df[lot_d]
        self.df['BLDAREA'] = self.df[bld_f] * self.df[bld_d]
        self.df['LTAREA'] = np.where(self.df['LTAREA'] == 0, self.df['BLDAREA'], self.df['LTAREA'])
        self.df['BLDAREA'] = np.where(self.df['BLDAREA'] == 0, self.df['LTAREA'], self.df['BLDAREA'])

        self.df.drop(columns=[lot_f, lot_d, bld_f, bld_d])

        self.fill_missing_column_values_with_mean_for_boro(self.df, 'LTAREA')
        self.fill_missing_column_values_with_mean_for_boro(self.df, 'BLDAREA')
