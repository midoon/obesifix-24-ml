import pandas as pd
import numpy as np
import sklearn
from sklearn.neighbors import NearestNeighbors


class Recommender:

    def __init__(self):
        self.data = pd.read_csv(
            'dataset/new_recommendation_processedfinal.csv')

    def get_features(self):
        # getting dummies of dataset
        preference_dummies = self.data.FoodCategory.str.get_dummies()
        cal_dummies = self.data.cal_standart.str.get_dummies(sep=' ')
        fat_dummies = self.data.fat_standart.str.get_dummies(sep=' ')
        carb_dummies = self.data.carbo_standart.str.get_dummies(sep=' ')
        protein_dummies = self.data.protein_standart.str.get_dummies(sep=' ')
        feature_df = pd.concat(
            [preference_dummies, cal_dummies, fat_dummies, carb_dummies, protein_dummies], axis=1)
        return feature_df

    def k_neighbor(self, inputs):
        feature_df = self.get_features()

        model = NearestNeighbors(n_neighbors=25, algorithm='ball_tree')

        # fitting model with dataset features
        model.fit(feature_df)

        df_results = pd.DataFrame(columns=list(self.data.columns))

        # getting distance and indices for k nearest neighbor
        distances, indices = model.kneighbors(inputs)

        for i in list(indices):
            df_results = pd.concat(
                [df_results, self.data.loc[i]], ignore_index=True)

        df_results = df_results.filter(['Name', 'Images', 'Calories', 'FatContent',
                                       'CarbohydrateContent', 'ProteinContent', 'Keywords', 'FoodCategory'])
        df_results = df_results.drop_duplicates(subset=['Name'])
        df_results = df_results.reset_index(drop=True)
        df_results = df_results.head(25)
        return df_results

    def recomend(nutrition_status, food_type):
        print(nutrition_status, food_type)
        # create variable for filtering
        ob = Recommender()
        data = ob.get_features()
        total_features = data.columns
        d = dict()
        for i in total_features:
            d[i] = 0
        # print(d)
        # input for preference and size
        pref_input = food_type
        pref_inputtt = pref_input.split(",")
        size_input = nutrition_status

        # variable change for every preference
        if size_input == 'obese':
            d['low_cal'] = 1
            d['low_fat'] = 1
            d['low_carb'] = 1
            d['low_pro'] = 1
        elif size_input == 'overweight':
            d['so_so_cal'] = 1
            d['low_fat'] = 1
            d['low_carb'] = 1
            d['low_pro'] = 1
        elif size_input == 'normal':
            d['midhigh_cal'] = 1
            d['so_so_fat'] = 1
            d['so_carb'] = 1
            d['so_so_pro'] = 1
        elif size_input == 'underweight':
            d['high_cal'] = 1
            d['high_fat'] = 1
            d['high_carb'] = 1
            d['high_pro'] = 1
        for i in pref_inputtt:
            d[i] = 1
        final_input = list(d.values())
        # print result
        results = ob.k_neighbor([final_input])
        return results
