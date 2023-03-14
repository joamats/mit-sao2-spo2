import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TKAgg')
import seaborn as sns
from sklearn.metrics import r2_score

from matplotlib.gridspec import GridSpec

df = pd.read_csv('data/MIMIC_IV.csv')[['SaO2','SpO2', 'race_group']]
df = df[(df.SaO2 <= 100) & (df.SpO2 <= 100) & (df.SaO2 >= 60) & (df.SpO2 >= 60)]

r2 = r2_score(df.SaO2, df.SpO2)

fig = plt.figure(figsize=(8,8))
gs = GridSpec(5,5)

ax_scatter = fig.add_subplot(gs[1:5, 0:4])

df_norm = df[~((df.SaO2 < 88) & (df.SpO2 >= 88))]
df_hypo = df[(df.SaO2 < 88) & (df.SpO2 >= 88)]

hh_perc = len(df_hypo)/(len(df_norm) + len(df_hypo))*100

sns.scatterplot(df_norm, x='SpO2', y='SaO2',
                marker='o', alpha=0.1, ax=ax_scatter)
sns.scatterplot(df_hypo, x='SpO2', y='SaO2',
                marker='o', color="tab:red", alpha=0.1, ax=ax_scatter,
                label="H.H")
ax_scatter.set_xlabel('SpO2 (%)')
ax_scatter.set_ylabel('SaO2 (%)')
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

fig.suptitle(f"SaO2 vs SpO2: R\u00B2 = {r2:.2f}, H.H = {hh_perc:.1f}%", fontsize=16, y=0.95)

fig.savefig('EDA/SaO2_SpO2.png', dpi=300, bbox_inches='tight')


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

