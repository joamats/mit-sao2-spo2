from utils import *
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TKAgg')
import seaborn as sns


df = pd.read_csv('data/MIMIC_IV.csv')[['SaO2','SpO2', 'race_group']]
df = df[(df.SaO2 <= 100) & (df.SpO2 <= 100) & (df.SaO2 >= 60) & (df.SpO2 >= 60)]

sao2_spo2_plot(df, 'EDA/SaO2vsSpO2')


fig2, ax2 = plt.subplots(2, 1, sharex= True, sharey=True, figsize=(5, 8))

races = ["White", "Black"]

for i, r in enumerate(races):

    sns.kdeplot(x=df[df.race_group == r].SaO2, ax=ax2[i], color='orange', fill=True, label="SaO2")
    sns.kdeplot(x=df[df.race_group == r].SpO2, ax=ax2[i], color='green', fill=True, label="SpO2")
    ax2[i].set_xlim([90,100])
    ax2[i].legend(loc='upper left', ncol=1, fontsize=9)
    ax2[i].set_xlabel('O2 Saturation (%)')
    ax2[i].set_title(f'{r} Patients')


fig2.suptitle('SaO2 and SpO2 Density Distributions')

fig2.savefig('EDA/SaO2vsSpO2.png', dpi=300, bbox_inches='tight')


fig3, ax3 = plt.subplots(1, 1, sharex= True, sharey=True, figsize=(8,4))

races = ["White", "Black"]
colors = ['orange', 'black']

for i, r in enumerate(races):

    sns.kdeplot(x=df[df.race_group == r].SpO2 - df[df.race_group == r].SaO2,
                ax=ax3, color=colors[i], fill=True, label=r)
    
ax3.set_xlim([-5,5])
ax3.legend(loc='upper left', ncol=1, fontsize=9)
ax3.set_xlabel('SpO2 - SaO2 (%)')
ax3.set_title('Difference between SpO2 and SaO2 across Race')

fig3.savefig('EDA/SaO2-SpO2.png', dpi=300, bbox_inches='tight')

