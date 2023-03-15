import pickle 
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error


def save_model(model, filename):
    """Save a model to a file"""
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

def load_model(filename):
    """Load a model from a file"""
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

# Adjusted R2
def adj_r2(r2, y_test, X_test):
    n = len(y_test)
    k = X_test.shape[1]
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)

def rmse(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

def evaluate_model(model, X_test, y_test, name):
    """Evaluate a model on a test set"""
    # Make predictions on the validation set
    y_pred = model.predict(X_test)

    # Compute evaluation metrics on Validation
    r2 = r2_score(y_test, y_pred)

    def create_results_df(y_test, X_test, y_pred, race):

        label, idxs = race[0], race[1] 

        # Filter the data according to race
        y_test = y_test.loc[idxs]
        X_test = X_test.loc[idxs]
        y_pred = y_pred[idxs]
    
        bas_r2 = r2_score(y_test, X_test.SpO2)
        bas_adj_r2 = adj_r2(bas_r2, y_test, X_test)
        bas_rmse = rmse(y_test, X_test.SpO2)
        b_hh = X_test[(y_test.SaO2 < 88) & (X_test.SpO2 >= 88)]
        bas_hh = len(b_hh)# / len(X_test)

        model_r2 = r2_score(y_test, y_pred)
        model_adj_r2 = adj_r2(model_r2, y_test, X_test)
        model_rmse = rmse(y_test, y_pred)
        m_hh = X_test[(y_test.SaO2 < 88) & (y_pred >= 88)]
        model_hh = len(m_hh)# / len(X_test)

        # Create a dataframe to store the results
        results = pd.DataFrame({'Race': [label],
                                'N': [len(y_test)],
                                'Baseline R\u00B2': [bas_r2],
                                'Baseline Adj R\u00B2': [bas_adj_r2],
                                'Baseline RMSE': [bas_rmse],
                                'Baseline HH': [bas_hh],
                                'Model R\u00B2': [model_r2],
                                'Model Adj R\u00B2': [model_adj_r2],
                                'Model RMSE': [model_rmse],
                                'Model HH': [model_hh],
                                'Delta R\u00B2': [model_r2 - bas_r2],
                                'Delta Adj R\u00B2': [model_adj_r2 - bas_adj_r2],
                                'Delta RMSE': [model_rmse - bas_rmse],
                                'Delta HH': [model_hh - bas_hh]
                                })
        
        return results

    races = {'Overall': [True] * len(y_test),
             'White': X_test.race_group == 1,
             'Hispanic': X_test.race_group == 2,
             'Asian': X_test.race_group == 3,
             'Other': X_test.race_group == 4,
             'Black': X_test.race_group == 5
             }

    results_list = []
    for r in races.items():
        results_list.append(create_results_df(y_test, X_test, y_pred, r))

    results = pd.concat(results_list)
    metric_order = results.columns[1:]
    race_order = races.keys()
    melted_df = results.melt(id_vars=['Race'], value_vars=metric_order, var_name='Metric')
    pivoted_df = melted_df.pivot(index=['Metric'], columns='Race', values='value')
    pivoted_df = pivoted_df[race_order].reindex(metric_order)

    pivoted_df.to_excel(f'results/{name}.xlsx')
    pivoted_df.to_latex(f'results/{name}.tex')
