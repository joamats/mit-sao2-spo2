import pandas as pd
import numpy as np

df_raw = pd.read_csv('data/MIMIC_IV.csv')

# keep only pairs within 10 min
df = df_raw[df_raw['delta_SpO2'] >= -10]

print("After Dropping Pairs not within 10 minutes")
print(f"No. Subjects: {df.subject_id.nunique()}")
print(df.race_group.value_counts()/len(df))
print(df.language.value_counts()/len(df))
print(df.gender.value_counts()/len(df))

# Clamp values to 65-100
df = df[(df.SaO2 <= 100) & (df.SpO2 <= 100) & (df.SaO2 >= 65) & (df.SpO2 >= 65)]

print("After Dropping Pairs out of 65-100")
print(f"No. Subjects: {df.subject_id.nunique()}")
print(df.race_group.value_counts()/len(df))
print(df.language.value_counts()/len(df))
print(df.gender.value_counts()/len(df))

# Drop features with high missingness, as reported in table 1 raw
df = df.drop(columns=['BMI', 'd_dimer', 'fibrinogen', 'thrombin',
                      'bilirubin_direct', 'bilirubin_indirect',
                      'ck_cpk', 'ck_mb', 'ggt', 'ld_ldh'])

# Encode Categorical Variables
df['language'] = df['language'].map({'ENGLISH': 1, '?': 0})
df['ventilation_status'] = df['ventilation_status'].map({'None': 0,
                                                         'SupplementalOxygen': 1,
                                                         'NonInvasiveVent': 2,
                                                         'HFNC': 2,
                                                         'InvasiveVent': 3,
                                                         'Tracheostomy': 4,
                                                         })

df['race_group'] = df['race_group'].map({'White': 1,
                                         'Hispanic': 2,
                                         'Asian': 3,
                                         'Other': 4,
                                         'Black': 5})

# mapping of categories
# normal (or near normal) rhythm: 0
# bradycardia with pacer: 1
# atrial dysrthymias: 2
# bundle branch blocks: 3
# ventricular dysrhythmias = very bad: 4

category_map = {'1st AV (First degree AV Block)': 0,
                '2nd AV M2 (Second degree AV Block - Mobitz 2)': 1,
                '2nd AV W-M1 (Second degree AV Block Wenckebach - Mobitz1)': 1,
                '3rd AV (Complete Heart Block)': 1,
                'A Flut (Atrial Flutter)': 2,
                'A Paced': 1,
                'AF (Atrial Fibrillation)': 2,
                'AV Paced': 1,
                'Asystole': 4,
                'Idioventricular': 3,
                'JR (Junctional Rhythm)': 3,
                'JT (Junctional Tachycardia)': 3,
                'LBBB (Left Bundle Branch Block)': 3,
                'MAT (Multifocal atrial tachycardia)': 2,
                'PAT (Paroxysmal Atrial Tachycardia)': 2,
                'RBBB (Right Bundle Branch Block)': 3,
                'SA (Sinus Arrhythmia)': 2,
                'SB (Sinus Bradycardia)': 0,
                'SR (Sinus Rhythm)': 0,
                'ST (Sinus Tachycardia)': 0,
                'SVT (Supra Ventricular Tachycardia)': 2,
                'V Paced': 3,
                'VF (Ventricular Fibrillation)': 4,
                'VT (Ventricular Tachycardia)': 4,
                'WAP (Wandering atrial pacemaker)': 1}

df['heart_rhythm'] = df['heart_rhythm'].map(category_map)
df['heart_rhythm'] = df['heart_rhythm'].fillna(0)

df['vasopressors'] = df.norepinephrine_equivalent_dose.apply(lambda x: 1 if x > 0 else 0)

# Replace nan with 0 in norepinephrine_equivalent_dose -> Baseline
df['norepinephrine_equivalent_dose'] = df['norepinephrine_equivalent_dose'].fillna(0)
# No ventilation information -> assume best case scenario
df['ventilation_status'] = df['ventilation_status'].fillna(0)

# one-hot encoding of FiO2 -> this way we avoid having to impute FiO2 for vent_status = 1
df['FiO2_0'] = df.apply(lambda row: 21          if row['ventilation_status'] == 0 else 0, axis=1)
df['FiO2_2'] = df.apply(lambda row: row['FiO2'] if row['ventilation_status'] == 2 else 0, axis=1)
df['FiO2_3'] = df.apply(lambda row: row['FiO2'] if row['ventilation_status'] == 3 else 0, axis=1)
df['FiO2_4'] = df.apply(lambda row: row['FiO2'] if row['ventilation_status'] == 4 else 0, axis=1)

# and same for these deltas
df['delta_FiO2_0'] = df.apply(lambda row: row['delta_FiO2'] if row['ventilation_status'] == 0 else 0, axis=1)
df['delta_FiO2_2'] = df.apply(lambda row: row['delta_FiO2'] if row['ventilation_status'] == 2 else 0, axis=1)
df['delta_FiO2_3'] = df.apply(lambda row: row['delta_FiO2'] if row['ventilation_status'] == 3 else 0, axis=1)
df['delta_FiO2_4'] = df.apply(lambda row: row['delta_FiO2'] if row['ventilation_status'] == 4 else 0, axis=1)

# df['sofa_resp_proxy'] = df.apply(lambda row: 0 if row['ventilation_status'] == 0 else(
#                                              1.5 if (row['ventilation_status'] == 1) else(
#                                              3 if ((row['ventilation_status'] >= 3) & (row['FiO2'] <= 50)) | (row['ventilation_status'] == 2) else(
#                                              4 if (row['ventilation_status'] >= 3)  & (row['FiO2'] > 50) else(
#                                              0)))), axis=1)

df.to_csv("data/MIMIC_IV_clean.csv")
