import pickle 
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import r2_score
from matplotlib.gridspec import GridSpec
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TKAgg')
import seaborn as sns

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

    races = {
             'Asian': X_test.race_group == 3,
             'Black': X_test.race_group == 5,
             'Hispanic': X_test.race_group == 2,
             'Other': X_test.race_group == 4,
             'White': X_test.race_group == 1,
             'Overall': [True] * len(y_test)
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

    pivoted_df.to_excel(f'results/{name}.xlsx', float_format='%.3f')
    pivoted_df.to_latex(f'results/{name}.tex', float_format='%.3f')

# Plot for the SaO2-SpO2 relationship maps
def sao2_spo2_plot(df, name, lbl_text):

    r2 = r2_score(df.SaO2, df.SpO2)

    fig = plt.figure(figsize=(8,8))
    gs = GridSpec(5,5)

    ax_scatter = fig.add_subplot(gs[1:5, 0:4])

    df_norm = df[~((df.SaO2 < 88) & (df.SpO2 >= 88))]
    df_hypo = df[(df.SaO2 < 88) & (df.SpO2 >= 88)]

    hh = len(df_hypo)
    tot = len(df_norm) + len(df_hypo)

    sns.scatterplot(df_norm, x='SpO2', y='SaO2',
                    marker='o', alpha=0.1, ax=ax_scatter)
    sns.scatterplot(df_hypo, x='SpO2', y='SaO2',
                    marker='o', color="tab:red", alpha=0.1, ax=ax_scatter,
                    label="H.H")
    ax_scatter.set_xlabel(f'{lbl_text} SpO2 (%)')
    ax_scatter.set_ylabel('SaO2 (%)')
    ax_scatter.set_xlim([64, 102])
    ax_scatter.set_ylim([64, 102])
    ax_scatter.grid(False)
    ax_scatter.get_legend().remove()

    ax_hist_x = fig.add_subplot(gs[0,0:4])
    sns.kdeplot(x=df.SpO2, ax=ax_hist_x, color='green', fill=True)
    ax_hist_x.xaxis.set_ticklabels([])
    ax_hist_x.grid(False)
    ax_hist_x.set_xlabel('')
    ax_hist_x.set_ylim([0,.35])
    ax_hist_x.axvline(x=df.SpO2.mean(), linewidth=0.8,
                    linestyle='--', color='green',
                    label="SpO2 Mean")


    ax_hist_y = fig.add_subplot(gs[1:5, 4])
    sns.kdeplot(y=df.SaO2, ax=ax_hist_y, color='orange', fill=True)
    ax_hist_y.yaxis.set_ticklabels([])
    ax_hist_y.grid(False)
    ax_hist_y.set_ylabel('')
    ax_hist_y.set_xlim([0,.35])
    ax_hist_y.axhline(y=df.SaO2.mean(), xmin=0, xmax=1,
                    linewidth=0.8, linestyle='--', color='orange',
                    label="SaO2 Mean")

    fig.legend(loc='upper right', bbox_to_anchor=(0.915, 0.85), ncol=1, fontsize=9)

    fig.suptitle(f"SaO2 vs SpO2: R\u00B2 = {r2:.2f}, H.H = {hh} / {tot}", fontsize=16, y=0.95)

    fig.savefig(f"{name}.png", dpi=300, bbox_inches='tight')