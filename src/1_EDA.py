from utils import *
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TKAgg')
import seaborn as sns
from tableone import TableOne

# Table 1

data = pd.read_csv('data/MIMIC_IV.csv')

data['race_white'] = data.race_group == 'White'

groupby = ['race_white']

categorical = ['hidden_hypoxemia', 'gender', 'race_group', 'language',
               'ventilation_status', 'invasive_vent', 'rrt',
               'heart_rhythm']

nonnormal = ['SaO2', 'delta_SpO2', 'SpO2', 'anchor_age', 'BMI',
             'los_hospital', 'los_icu', 'CCI', 'SOFA_admission',
             'sofa_coag', 'sofa_liver', 'sofa_cv',
             'sofa_cns', 'sofa_renal', 'FiO2',
             'norepinephrine_equivalent_dose', 
             'hemoglobin', 'hematocrit', 'mch', 'mchc', 'mcv', 'platelet',
             'rbc', 'rdw', 'wbc', 'd_dimer', 'fibrinogen',
             'thrombin', 'inr', 'pt', 'ptt', 'alt', 'alp', 'ast',
             'bilirubin_total', 'bilirubin_direct', 'bilirubin_indirect',
             'ck_cpk', 'ck_mb', 'ggt', 'ld_ldh', 'albumin', 'aniongap',
             'bicarbonate', 'bun', 'calcium', 'chloride', 'creatinine',
             'glucose_lab', 'sodium', 'potassium', 'ph', 'lactate',
             'heart_rate', 'mbp', 'resp_rate', 'temperature', 'glucose']

# Create a TableOne 

table1_raw = TableOne(data, columns=categorical+nonnormal,
                      groupby=groupby, categorical=categorical, nonnormal=nonnormal,
                      smd=True, 
                      dip_test=True, normal_test=True, tukey_test=True, htest_name=True)

table1_raw.to_excel('EDA/raw_table1.xlsx')

# Plots
df = pd.read_csv('data/MIMIC_IV.csv')[['SaO2','SpO2','delta_SpO2', 'race_group']]
df = df[(df.SaO2 <= 100) & (df.SpO2 <= 100) & (df.SaO2 >= 65) & (df.SpO2 >= 65)]
df = df[df.delta_SpO2 >= -10]

sao2_spo2_plot(df, 'EDA/map_SaO2vsSpO2', 'Measured')

# Modify Race Group to be White or Non-White and compare there
df.race_group = df.race_group.apply(lambda x: 'Non-White' if x != 'White' else 'White')

fig2, ax2 = plt.subplots(2, 1, sharex= True, sharey=True, figsize=(5, 8))

races = ["White", "Non-White"]

for i, r in enumerate(races):

    sns.kdeplot(x=df[df.race_group == r].SaO2, ax=ax2[i], color='orange', fill=True, label="SaO2")
    sns.kdeplot(x=df[df.race_group == r].SpO2, ax=ax2[i], color='green', fill=True, label="SpO2")
    ax2[i].set_xlim([90,100])
    ax2[i].legend(loc='upper left', ncol=1, fontsize=9)
    ax2[i].set_xlabel('O2 Saturation (%)')
    ax2[i].set_title(f'{r} Patients')

fig2.suptitle('SaO2 and SpO2 Density Distributions')

fig2.savefig('EDA/dist_SaO2vsSpO2.png', dpi=300, bbox_inches='tight')


colors = ['green', 'black']
fig3, ax3 = plt.subplots(1, 1, sharex= True, sharey=True, figsize=(8,4))

for i, r in enumerate(races):

    sns.kdeplot(x=df[df.race_group == r].SpO2 - df[df.race_group == r].SaO2,
                ax=ax3, color=colors[i], fill=True, label=r)
    
ax3.set_xlim([-4,4])
ax3.legend(loc='upper left', ncol=1, fontsize=9)
ax3.set_xlabel('SpO2 - SaO2 (%)')
ax3.set_title('SpO2 - SaO2 Gap Distribution, White vs. Non-White Patients')

fig3.savefig('EDA/dist_SaO2-SpO2.png', dpi=300, bbox_inches='tight')

