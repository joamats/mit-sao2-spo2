import pandas as pd

df = pd.read_csv('data/MIMIC_IV.csv')

# Drop features with high missingness, as reported in table 1 raw
df = df.drop(columns=['BMI', 'd_dimer', 'fibrinogen', 'thrombin',
                      'bilirubin_direct', 'bilirubin_indirect',
                      'ck_cpk', 'ck_mb', 'ggt', 'ld_ldh'])

# Replace nan with 0 when appropriate (SOFA and Treatments)
df['sofa_coag'] = df['sofa_coag'].fillna(0)
df['sofa_liver'] = df['sofa_liver'].fillna(0)
df['sofa_cv'] = df['sofa_cv'].fillna(0)
df['sofa_cns'] = df['sofa_cns'].fillna(0)
df['sofa_renal'] = df['sofa_renal'].fillna(0)
##### missing sofa respiratory

df['norepinephrine_equivalent_dose'] = df['norepinephrine_equivalent_dose'].fillna(0)
df['ventilation_status'] = df['ventilation_status'].fillna(0)

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

# Let's keep just the first pair per patient (it's been sorted by subject_id, stay_id,time)
df_final = df.groupby('subject_id').first().reset_index()

print(f"{(len(df) - len(df_final))/len(df)*100:.2f}% rows dropped \
      \n{len(df_final)} rows remaining")

df.to_csv("data/MIMIC_IV_clean.csv")
