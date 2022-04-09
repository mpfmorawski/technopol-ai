import pandas as pd
import numpy as np
import geocoder
import requests
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class Preprocessing(object):
    """
    Class for first preprocessing of given data
    """

    def __init__(self, path, boro = False, standardization = True):
        self.df = self.read_data(path)
        self.df_continous = pd.DataFrame()
        self.df_categorical = pd.DataFrame()
        self.df_value = pd.DataFrame()
        self.boro = boro
        self.standardization = standardization

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
                    if g is not None:
                        self.df.loc[index, 'POSTCODE'] = g.postal
                        self.df.loc[index, 'Latitude'] = g.lat
                        self.df.loc[index, 'Longitude'] = g.lng
                        print(index)

    def get_mean_column_value_for_boro(self, data, column_name: str, boro: int, include_zeros: bool = False):
        column_vals = data[data['BORO'] == boro][column_name]
        if include_zeros:
            return np.mean(column_vals)
        else:
            return np.mean(column_vals[column_vals != 0])

    def fill_missing_column_values_with_mean_for_boro(self, data, column_name: str, replace_zeros: bool = False):
        mean_area_per_boro = {}
        for boro in data['BORO'].unique():
            mean_area_per_boro[boro] = self.get_mean_column_value_for_boro(data, column_name, boro)

        for i, row in data.iterrows():
            if pd.isna(row[column_name]) or (replace_zeros and row[column_name] == 0):
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

        self.df = self.df[self.df['FULLVAL'] < 25000000]

    def split_into_continous_and_categorical_data(self):
        self.df_continous = self.df[['STORIES', 'AVLAND', 'AVTOT', 'EXLAND', 'EXTOT', 'AVLAND2', 'AVTOT2', 'EXLAND2', 'EXTOT2', 'Latitude', 'Longitude', 'LTAREA', 'BLDAREA']]
        self.df_categorical =  self.df[['BORO', 'BLOCK', 'LOT', 'EASEMENT', 'BLDGCL', 'TAXCLASS', 'EXT', 'STADDR', 'POSTCODE', 'EXMPTCL', 'EXCD2', 'YEAR', 'VALTYPE', 'Community Board', 'Council District', 'Census Tract', 'BIN', 'NTA', ]]
        self.df_value = self.df[['FULLVAL']]
        self.corr_df = self.df[['FULLVAL', 'STORIES', 'AVLAND', 'AVTOT', 'EXLAND', 'EXTOT', 'AVLAND2', 'AVTOT2', 'EXLAND2', 'EXTOT2', 'Latitude', 'Longitude', 'LTAREA', 'BLDAREA']]

    def split_data_into_tax_categories(self, category):
        self.df = self.df[self.df['TAXCLASS'].isin(category)]

    def split_data_into_tax_boro_categories(self, category, boro):
        self.df = self.df[(self.df['TAXCLASS'].isin(category)) & (self.df['BORO'] == int(boro))]

    def caluclate_coefficient(self, treshold = 0.7):
        correlation =self.corr_df.corr(method="pearson")
        corr = pd.DataFrame(correlation['FULLVAL'])
        print(corr)
        columns_filtered = list(corr[corr['FULLVAL'].abs() >= treshold].index)
        columns_filtered.remove('FULLVAL')
        return columns_filtered

    def remove_NaN_column(self, column_list):
        column_list.append('Latitude')
        #column_list.append('STORIES')
        df_new = self.df[column_list]
        df_new = df_new.dropna(axis=1)
        return df_new

    def apply_scaling_on_continuous(self):
        self.scaler_std_x = StandardScaler()
        column_names = list(self.df_continous.columns.values)
        self.df_continous = self.scaler_std_x.fit_transform(self.df_continous)
        self.df_continous = pd.DataFrame(self.df_continous, columns=column_names)

        # self.scaler_std_y = StandardScaler()
        # column_names = list(self.df_value.columns.values)
        # self.df_value = self.scaler_std_y.fit_transform(self.df_value)
        # self.df_value = pd.DataFrame(self.df_value, columns=column_names)

    def run(self, treshold, category, boro = ""):
        self.trim_redundant_data()
        #self.fill_missing_geodata()
        self.add_columns_with_distances_to_main_places()
        self.replace_lengths_with_areas()

        self.fill_missing_column_values_with_mean_for_boro(self.df, 'Latitude', replace_zeros=True)
        self.fill_missing_column_values_with_mean_for_boro(self.df, 'Longitude', replace_zeros=True)

        for col_name in ['AVLAND2', 'AVTOT2', 'EXLAND2', 'EXTOT2']:
            self.df[col_name] = self.df[col_name].fillna(0)

        if self.boro :
            self.split_data_into_tax_boro_categories(category, boro)
        else :
            self.split_data_into_tax_categories(category)

        self.split_into_continous_and_categorical_data()
        if self.standardization:
            self.apply_scaling_on_continuous()
        columns_name = self.caluclate_coefficient(treshold)
        filtered_df = self.remove_NaN_column(columns_name)
        return filtered_df, self.df_value
