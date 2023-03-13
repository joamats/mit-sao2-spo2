from tableone import TableOne
import pandas as pd

data = pd.read_csv('data/MIMIC_IV.csv')

data['race_white'] = data.race_group == 'White'

groupby = ['race_white']

categorical = ['hidden_hypoxemia', 'gender', 'race_group', 'language',
               'ventilation_status', 'invasive_vent', 'rrt']
               #'heart_rhythm']

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

table1_raw.to_excel('EDA/table1_raw.xlsx')