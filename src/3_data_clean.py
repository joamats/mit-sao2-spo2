import pandas as pd

df = pd.read_csv('data/MIMIC_IV.csv')

# Clamp values to 60-100
df = df[(df.SaO2 <= 100) & (df.SpO2 <= 100) & (df.SaO2 >= 65) & (df.SpO2 >= 65)]

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

category_column = df['heart_rhythm']
df['heart_rhythm'] = category_column.map(category_map)

df['vasopressors'] = df.norepinephrine_equivalent_dose.apply(lambda x: 1 if x > 0 else 0)

# Replace nan with 0 in SOFA -> Assuming best case scenario
df['sofa_coag'] = df['sofa_coag'].fillna(0)
df['sofa_liver'] = df['sofa_liver'].fillna(0)
df['sofa_cv'] = df['sofa_cv'].fillna(0)
df['sofa_cns'] = df['sofa_cns'].fillna(0)
df['sofa_renal'] = df['sofa_renal'].fillna(0)
df['sofa_resp'] = df['sofa_resp'].fillna(0)

# Replace nan with 0 in norepinephrine_equivalent_dose -> Baseline
df['norepinephrine_equivalent_dose'] = df['norepinephrine_equivalent_dose'].fillna(0)
# No ventilation information -> assume best case scenario
df['ventilation_status'] = df['ventilation_status'].fillna(0)
# No FiO2 information -> assume room air
df['FiO2'] = df['FiO2'].fillna(21) # Room Air O2 %

# Let's keep just the first pair per patient (it's been sorted by subject_id, stay_id,time)
df_final = df#.groupby('subject_id').first().reset_index()

print(f"{(len(df) - len(df_final))/len(df)*100:.2f}% rows dropped \
      \n{len(df_final)} rows remaining")

df_final.to_csv("data/MIMIC_IV_clean.csv")
