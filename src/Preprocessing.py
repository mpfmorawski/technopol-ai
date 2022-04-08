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
        self.df = self.df.drop(columns = columns_to_drop)

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

    def run(self):
        self.trim_redundant_data()
        self.add_columns_with_distances_to_main_places()
        return self.df
