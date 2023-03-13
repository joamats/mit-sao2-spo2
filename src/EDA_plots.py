import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TKAgg')
import seaborn as sns

from matplotlib.gridspec import GridSpec

df = pd.read_csv('data/MIMIC_IV.csv')[['SaO2','SpO2']]
df = df[(df.SaO2 <= 100) & (df.SpO2 <= 100) & (df.SaO2 >= 60) & (df.SpO2 >= 60)]

fig = plt.figure(figsize=(8,8))
gs = GridSpec(5,5)

ax_scatter = fig.add_subplot(gs[1:5, 0:4])

sns.scatterplot(df, x='SpO2', y='SaO2', marker='o', alpha=0.1, ax=ax_scatter)
ax_scatter.set_xlabel('SpO2 (%)')
ax_scatter.set_ylabel('SaO2 (%)')
ax_scatter.grid(False)

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
sns.kdeplot(y=df.SaO2, ax=ax_hist_y, color='red', fill=True)
ax_hist_y.yaxis.set_ticklabels([])
ax_hist_y.grid(False)
ax_hist_y.set_ylabel('')
ax_hist_y.set_xlim([0,.35])
ax_hist_y.axhline(y=df.SaO2.mean(), xmin=0, xmax=1,
                  linewidth=0.8, linestyle='--', color='red',
                  label="SaO2 Mean")

fig.legend(loc='upper right', bbox_to_anchor=(0.915, 0.85), ncol=1, fontsize=9)

fig.suptitle("SaO2 vs SpO2 Distributions", fontsize=16, y=0.95)

fig.savefig('EDA/SaO2_SpO2.png', dpi=300, bbox_inches='tight')


fig2, ax2 = plt.subplots(1,1, figsize=(6, 4))

sns.kdeplot(x=df.SaO2, ax=ax2, color='red', fill=True, label="SaO2")
sns.kdeplot(x=df.SpO2, ax=ax2, color='green', fill=True, label="SpO2")
ax2.set_xlim([90,100])
ax2.legend(loc='upper left', ncol=1, fontsize=9)
ax2.set_xlabel('O2 Saturation (%)')
ax2.set_title('SaO2 and SpO2 Density Distributions')

fig2.savefig('EDA/SaO2vsSpO2.png', dpi=300, bbox_inches='tight')

