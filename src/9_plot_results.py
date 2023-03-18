import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TKAgg')

def plot_results(model_name, title):

    # load the data
    data = pd.read_excel(f'results/{model_name}.xlsx', index_col=0)
    data = data.drop('Overall', axis=1)

    xx = np.arange(len(data.columns))

    # create the first subplot with baseline and model R²
    fig, axs = plt.subplots(1,3, figsize=(14, 4))
    axs[0].bar(xx-.3, data.loc['Baseline R²'] * 100,
            width=.3, 
            color='lightgray', label='Baseline')

    axs[0].bar(xx, data.loc['Model R²'] * 100,
            width=.3, #alpha=.8,
            color='mediumseagreen', label='Model')

    axs[0].set_title('SaO2 vs. SpO2 Goodness of Fit')
    axs[0].set_xticks(xx, data.columns)
    axs[0].set_xlabel("Race Group")
    axs[0].set_ylabel('R² (%)')
    axs[0].set_ylim([-100, 100])
    axs[0].axhline(y=0, color='k', linewidth=.7)


    # create the second subplot with baseline and model RMSE
    axs[1].bar(xx-.3, data.loc['Baseline RMSE'],
            width=.3,
            color='lightgray', label='Baseline')

    axs[1].bar(xx, data.loc['Model RMSE'],
            width=.3,
            color='mediumseagreen', label='Model')

    axs[1].set_title('Error in SaO2 vs. SpO2 values')
    axs[1].set_xticks(xx, data.columns)
    axs[1].set_xlabel("Race Group")
    axs[1].set_ylabel('RMSE (absolute change in %)')
    axs[1].set_ylim([0, 5])
    plt.xticks(xx, data.columns)


    # create the second subplot with baseline and model HH
    axs[2].bar(xx-.3, data.loc['Baseline HH'] / data.loc['N'] * 100,
            width=.3,
            color='lightgray', label='Baseline')

    axs[2].bar(xx, data.loc['Model HH'] / data.loc['N'] * 100,
            width=.3,
            color='mediumseagreen', label='Model')

    axs[2].set_title('Presence of Hidden Hypoxemias, H.H.')
    axs[2].set_xticks(xx, data.columns)
    axs[2].set_xlabel("Race Group")
    axs[2].set_ylabel('H.H. (%)')
    axs[2].set_ylim([0, 4])
    plt.xticks(xx, data.columns)
    axs[2].legend(loc='upper left', bbox_to_anchor=(1.05, .6))

    # add a title and save the figure
    plt.suptitle(f'SaO2 - SpO2 Correction Model{title}', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'results/{model_name}.png', dpi=300)


model_names = ["xgbr_wSOFA", "xgbr_woSOFA"]
titles = [', with SOFA Respiratory', ', without SOFA Respiratory']

for m, t in zip(model_names, titles):
    plot_results(m,t)
