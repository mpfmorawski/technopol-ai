import pandas as pd
import numpy as np
import geocoder
import requests


class Preprocessing(object):
    """
    Class for first preprocessing of given data
    """

    def __init__(self, path):
        self.df = self.read_data(path)

    def read_data(self, path):
        return pd.read_csv(path, on_bad_lines="skip")

    def trim_redundant_data(self):
        columns_to_drop = [
            'BBLE',
            'OWNER',
            'PERIOD',
            'Borough',
            'New Georeferenced Column'
        ]
        self.df = self.df.drop(columns=columns_to_drop)

    def add_lat_to_central_park_column(self):
        central_park_lat = 40.785091
        self.df['lat_to_central_park'] = self.df['Latitude'] - central_park_lat

    def add_long_to_central_park_column(self):
        central_park_long = -73.968285
        self.df['long_to_central_park'] = self.df['Longitude'] - central_park_long

    def add_lat_to_times_square_column(self):
        times_square_lat = 40.758896
        self.df['lat_to_times_square'] = self.df['Latitude'] - times_square_lat

    def add_long_to_times_square_column(self):
        times_square_long = -73.985130
        self.df['long_to_times_square'] = self.df['Longitude'] - times_square_long

    def add_lat_to_financial_district_column(self):
        financial_district_lat = 40.7075
        self.df['lat_to_financial_district'] = self.df['Latitude'] - financial_district_lat

    def add_long_to_financial_district_column(self):
        financial_district_long = -74.009167
        self.df['long_to_financial_district'] = self.df['Longitude'] - financial_district_long

    def add_columns_with_distances_to_main_places(self):
        self.add_lat_to_central_park_column()
        self.add_long_to_central_park_column()
        self.add_lat_to_times_square_column()
        self.add_long_to_times_square_column()
        self.add_lat_to_financial_district_column()
        self.add_long_to_financial_district_column()

    def fill_missing_geodata(self):
        with requests.Session() as session:
            for index, row in self.df.iterrows():
                if pd.isnull(row['POSTCODE']) and not pd.isnull(row['STADDR']):
                    g = geocoder.osm(row['STADDR'] + ', New York, NY', session=session)
                    if (g is not None):
                        row['POSTCODE'] = g.postal
                        row['Latitude'] = g.lat
                        row['Longitude'] = g.lng
                        print(index)

    def run(self):
        self.trim_redundant_data()
        self.fill_missing_geodata()
        self.add_columns_with_distances_to_main_places()
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
