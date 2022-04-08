import pandas as pd
import geocoder
import requests

class Preprocessing(object):
    """
    Class for first preprocessing of given data
    """

    def __init__(self, path):
        self.df = self.read_data(path)

    def read_data(self, path):
        return pd.read_csv(path, error_bad_lines=False)

    def trim_redundant_data(self):
        columns_to_drop = [
            'BBLE',
            'OWNER',
            'PERIOD',
            'Borough',
            'New Georeferenced Column'
        ]
        return self.df.drop(columns=columns_to_drop)

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
        self.df = self.trim_redundant_data()
        self.fill_missing_geodata()
        return self.df
