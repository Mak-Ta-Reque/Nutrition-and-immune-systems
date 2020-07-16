import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
import numpy as np

# Define Input of model
INPUT = ['Obesity', 'Undernourished', 'Deaths', 'Recovered', 'Active', 'Confirmed']
OUTPUT = ['Animal Products', 'Animal fats', 'Meat', 'Cereals - Excluding Beer','Vegetables','Vegetal Products','Vegetable Oils', 'Oilcrops' ]
'''['Alcoholic Beverages', 'Animal Products', 'Animal fats',
          'Aquatic Products, Other', 'Cereals - Excluding Beer', 'Eggs',
          'Fish, Seafood', 'Fruits - Excluding Wine', 'Meat',
          'Milk - Excluding Butter', 'Offals', 'Oilcrops', 'Pulses', 'Spices',
          'Starchy Roots', 'Stimulants', 'Sugar Crops', 'Sugar & Sweeteners',
          'Treenuts', 'Vegetal Products', 'Vegetable Oils', 'Vegetables',
          'Miscellaneous']'''

top8output = ['Animal Products', 'Animal fats', 'Meat', 'Cereals - Excluding Beer','Vegetables','Vegetal Products','Vegetable Oils', 'Oilcrops' ]
top23output = ['Alcoholic Beverages', 'Animal Products', 'Animal fats',
          'Aquatic Products, Other', 'Cereals - Excluding Beer', 'Eggs',
          'Fish, Seafood', 'Fruits - Excluding Wine', 'Meat',
          'Milk - Excluding Butter', 'Offals', 'Oilcrops', 'Pulses', 'Spices',
          'Starchy Roots', 'Stimulants', 'Sugar Crops', 'Sugar & Sweeteners',
          'Treenuts', 'Vegetal Products', 'Vegetable Oils', 'Vegetables',
          'Miscellaneous']

DATA_PATH = 'data'


# dataset class preprocess the data
class NutritionData:

    def __init__(self, data_path=None):

        self.GOOD_OUTPUT = OUTPUT
        self.INPUT = INPUT
        model_path = os.path.join(os.path.split(data_path)[0], 'models')
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        self.model_path = os.path.join('data/models', str(len(self.GOOD_OUTPUT)) + 'output')
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.prefix_name = os.path.split(data_path)[-1].split('.')[0]
        self.data_frame = pd.read_csv(data_path, index_col='Country')
        self.NOT_IMPORTANT_COLUMNS = pd.Index(self.data_frame.columns.values.tolist())
        self.NOT_IMPORTANT_COLUMNS = self.NOT_IMPORTANT_COLUMNS.drop(self.GOOD_OUTPUT + self.INPUT)
        #print(self.NOT_IMPORTANT_COLUMNS)

        data_frame = self.__replaceNaN(self.data_frame, replaceby="mostfrq")
        clm = data_frame.columns.drop(self.NOT_IMPORTANT_COLUMNS.append(pd.Index(self.INPUT)))

        data_frame = self.__replace_char(data_frame, "Undernourished", char=r'^<')
        scaled_df = self.__scale(data_frame, clm, scalar="mm")
        scaled_df = scaled_df.sample(frac=1, random_state=1).reset_index(drop=True)
        self.dataset = scaled_df

        # Importan Methodes

    # Replace the nan vale with mean, median, or most frequent value
    def __replaceNaN(self, dataframe, replaceby="mean"):
        if replaceby == "mean":
            return dataframe.fillna(dataframe.mean())

        elif replaceby == "median":
            return dataframe.fillna(dataframe.median())

        elif (replaceby == "mostfrq"):
            return dataframe.fillna(dataframe.mode().iloc[0])

        else:
            return dataframe.dropna()

    # Replce any nonnumaric char from a column.
    def __replace_char(self, dataframe, collumn='Undernourished', char=r'^<'):

        dataframe[collumn] = dataframe[collumn].replace(regex=char, value='')
        dataframe[collumn] = pd.to_numeric(dataframe[collumn], errors='coerce')
        return dataframe

    # Scale the data using standard scaler or max min scaler. Menstion the names to scale.
    def __scale(self, dataframe, coulumns, scalar="ss"):

        if scalar == "mm":
            from sklearn.preprocessing import MinMaxScaler
            scalar = MinMaxScaler()
            scalar.fit(dataframe[coulumns])
            dataframe[coulumns] = scalar.fit_transform(dataframe[coulumns])
            self.scalar_path = os.path.join(self.model_path, self.prefix_name + '_' + 'scalar.pkl')
            with open(self.scalar_path, 'wb') as file:
                pickle.dump(scalar, file)
            return dataframe

        elif scalar == "ss":
            from sklearn.preprocessing import StandardScaler
            scalar = StandardScaler()
            scalar.fit(dataframe[coulumns])
            dataframe[coulumns] = scalar.fit_transform(dataframe[coulumns])
            with open(os.path.join(self.model_path, self.prefix_name + '_' + 'scalar.pkl'), 'wb') as file:
                pickle.dump(scalar, file)
            return dataframe

        else:
            return dataframe


# NutritionData("data/Fat_Supply_Quantity_Data.csv")

# Generalizing class to test and train all the models
class NutritionModel:

    def __init__(self, dataset=None):
        self.dataset = dataset
        self.model = None
        self.train_set = None
        self.test_set = None

    def split_data(self, test_size=0.2, random_state=42):
        # splits the dataset in train and test set
        self.train_set, self.test_set = train_test_split(self.dataset, test_size=test_size, random_state=42)


    def set_model(self, model):
        # setting up the model
        self.model = model

    def save_model(self, pkl_filename='model.pkl'):
        with open(pkl_filename, 'wb') as file:
            pickle.dump(self.model, file)

    def load_model(self, pkl_filename='model.pkl'):
        with open(pkl_filename, 'rb') as file:
            self.model = pickle.load(file)

    def train_model(self, x_column, y_column):
        # traing the model
        # input arg: x column name, y column names
        self.model.fit(self.train_set[x_column], self.train_set[y_column])

    def test_model(self, x_column, y_column):
        score = self.model.score(self.test_set[x_column], self.test_set[y_column])
        return score

    def predict(self, X, scalar):
        with open(scalar, 'rb') as file:
            scalar = pickle.load(file)
        return scalar.inverse_transform(self.model.predict(X))


# Set all possible hyper parameters for the ML algorithms


from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import RandomizedSearchCV

method_names = {}
# Adding linear regression
LinearREG = LinearRegression()
method_names["Linear Regression"] = LinearREG

# Adding reage regression
params_Ridge = {'alpha': [1, 0.1, 0.01, 0.001, 0.0001, 0],
                "fit_intercept": [True, False],
                "solver": ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']}

LinearREGRedgeCV = RandomizedSearchCV(estimator=Ridge(),
                                      param_distributions=params_Ridge,
                                      n_iter=10,
                                      cv=5, verbose=2,
                                      random_state=42,
                                      n_jobs=-1, refit=True)

method_names["Redge"] = LinearREGRedgeCV

# Adding Lasso regression

params_Lasso = {'alpha': [1, 0.1, 0.01, 0.001, 0.0001, 0],
                "fit_intercept": [True, False],
                }
LinearREGLassoCV = RandomizedSearchCV(estimator=Lasso(),
                                      param_distributions=params_Lasso,
                                      n_iter=10,
                                      cv=5, verbose=2,
                                      random_state=42,
                                      n_jobs=-1, refit=True)

method_names["Lasso"] = LinearREGLassoCV

# Adding SVR


params_SVR = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
               'C': [1, 10, 100, 1000]},
              {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

SupportVectorRegression = RandomizedSearchCV(estimator=SVR(),
                                             param_distributions=params_SVR,
                                             n_iter=10,
                                             cv=5, verbose=2,
                                             random_state=42,
                                             n_jobs=-1, refit=True)

method_names["SVR"] = SupportVectorRegression

# Adding Decission Tree
params_decission_tree = [{'criterion': ['mse', 'friedman_mse', 'mae'],
                          'ccp_alpha': [0.0, 1e-3, 1e-4, 0.5, 1.0],
                          'max_features': [1, 2, 3, 4, 5],
                          'max_depth': [1, 2, 3, 4, ],
                          'random_state': [1, 4, 5, 100]},
                         {'criterion': ['friedman_mse'],
                          'ccp_alpha': [1.0],
                          'max_features': [1, 2, 3, 4, 5],
                          'max_depth': [10, 11, 12, 13, 14],
                          'random_state': [1, 4, 5, 100]},
                         {'criterion': ['mae'],
                          'ccp_alpha': [0.0, 1e-3, 1e-4, 0.5, 1.0],
                          'max_features': [1, 2, 3, 4, 5],
                          'max_depth': [5, 6, 7, 8, 9],
                          'random_state': [100]},

                         ]

# print(DecisionTreeRegressor().get_params().keys())

DecissionTree = RandomizedSearchCV(estimator=DecisionTreeRegressor(),
                                   param_distributions=params_decission_tree,
                                   n_iter=10,
                                   cv=5, verbose=2,
                                   random_state=42,
                                   n_jobs=-1, refit=True)

method_names["DecissionTree"] = DecissionTree

#  Adding RandomForestRegressor


params_random_forest = {'n_estimators': [int(x) for x in np.linspace(start=200, stop=2000, num=10)],
                        'max_features': ['auto', 'sqrt'],
                        'max_depth': [int(x) for x in np.linspace(10, 110, num=11)],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4],
                        'bootstrap': [True, False]}

RandomForest = RandomizedSearchCV(estimator=RandomForestRegressor(),
                                  param_distributions=params_random_forest,
                                  n_iter=10,
                                  cv=5, verbose=2,
                                  random_state=42,
                                  n_jobs=-1, refit=True)

method_names["RandomForest"] = RandomForest


def search_models(data_file):
    data_obj = NutritionData(data_path=data_file)
    model = NutritionModel(dataset=data_obj.dataset)
    model.split_data()
    x_val = data_obj.INPUT
    predictors = data_obj.GOOD_OUTPUT
    r2_score = []
    best_param = []
    for key, value in method_names.items():
        model.set_model(model=value)
        if key == 'SVR':
            score = 0
            for item in predictors:
                model.train_model(x_column=x_val, y_column=item)
                score += model.test_model(x_column=x_val, y_column=item)
                # print(score/len(GOOD_OUTPUT))
            r2_score.append(score / len(predictors))
            best_param.append(model.model.get_params())

        else:
            model.train_model(x_column=x_val, y_column=predictors)
            score = model.test_model(x_column=x_val, y_column=predictors)
            r2_score.append(score)
            best_param.append(model.model.get_params())
            # print(score)

    data_frame_dict = {'Methods': list(method_names.keys()),
                       'R^2 Score': r2_score,
                       'best_parameters': best_param}

    df = pd.DataFrame(data=data_frame_dict)
    return df


# Store the best model for all datasets

def store_models(data_file, method_name):
    dataobj = NutritionData(data_path=data_file)
    model = NutritionModel(dataset=dataobj.dataset)
    model.split_data()
    x_val = dataobj.INPUT
    predictors = dataobj.GOOD_OUTPUT
    model.set_model(model=method_names[method_name])
    model_file = os.path.join(dataobj.model_path, os.path.split(data_file)[-1].split('.')[0] + '.pkl')

    if method_name == 'SVR':
        score = 0
        for item in predictors:
            model.train_model(x_column=x_val, y_column=item)
            score += model.test_model(x_column=x_val, y_column=item)
            print(score / len(predictors))

    else:
        model.train_model(x_column=x_val, y_column=predictors)
        score = model.test_model(x_column=x_val, y_column=predictors)
        ideal_prediction = model.predict([[0, 0, 0, 1, 0.2, 0.2], ], dataobj.scalar_path)
        print(ideal_prediction)
        # ideal_prediction = dataobj.scaler.inverse_transform(ideal_prediction)
        model.save_model(model_file)

        # print(score)

    return model


pd.set_option("display.max_rows", None, "display.max_columns", None)


# print(search_models('data/Fat_Supply_Quantity_Data.csv').head())

# store_models('data/Fat_Supply_Quantity_Data.csv', 'DecissionTree')


def search_four_models(data_file_dir):
    data_csv_paths = [os.path.join(data_file_dir, "Food_Supply_Quantity_kg_Data.csv"),
                      os.path.join(data_file_dir, "Fat_Supply_Quantity_Data.csv"),
                      os.path.join(data_file_dir, "Food_Supply_kcal_Data.csv"),
                      os.path.join(data_file_dir, "Protein_Supply_Quantity_Data.csv")]
    # We will store all model performance related data there
    output_csv_path = os.path.join(data_file_dir, 'models_performance')
    if not os.path.exists(output_csv_path):
        os.makedirs(output_csv_path)

    for p in data_csv_paths:
        protein_prediction = search_models(p)
        # Storing based on the amout of output

        feature_related_path = os.path.join(output_csv_path, str(len(OUTPUT)) + 'predictors')
        if not os.path.exists(feature_related_path):
            os.makedirs(feature_related_path)

        output_path = os.path.join(feature_related_path, os.path.split(p)[-1])
        protein_prediction.to_csv(output_path)


def store_four_models(data_file_dir, method):
    data_csv_paths = [os.path.join(data_file_dir, "Food_Supply_Quantity_kg_Data.csv"),
                      os.path.join(data_file_dir, "Fat_Supply_Quantity_Data.csv"),
                      os.path.join(data_file_dir, "Food_Supply_kcal_Data.csv"),
                      os.path.join(data_file_dir, "Protein_Supply_Quantity_Data.csv")]
    # We will store all model performance related data there

    for p in data_csv_paths:
        model = store_models(p, method)

        # Storing based on the amout of output


#search_four_models('data')


def predict(scaler_name, model_name_pkl, input_data):
    model = NutritionModel()
    model.load_model(model_name_pkl)
    return model.predict(input_data, scaler_name)


#output = predict('data/models/2output/Protein_Supply_Quantity_Data_scalar.pkl',
#                 'data/models/2output/Protein_Supply_Quantity_Data.pkl',
#                 [[0, 0, 0, 1, 0.5, 0.5]])

#print(output)

#store_four_models('data', 'RandomForest')