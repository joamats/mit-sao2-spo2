import pandas as pd
import numpy as np

# load data
data = pd.read_csv('data/MIMIC_IV_clean.csv')

# Replace nan with 0 in deltas -> no information 
data['delta_vent_start'] = data['delta_vent_start'].fillna(0)
data['delta_rrt'] = data['delta_rrt'].fillna(0)
data['delta_vp_start'] = data['delta_vp_start'].fillna(0)

# Replace nan with 0 in SOFA -> Assuming best case scenario
data['sofa_coag'] = data['sofa_coag'].fillna(0)
data['sofa_liver'] = data['sofa_liver'].fillna(0)
data['sofa_cv'] = data['sofa_cv'].fillna(0)
data['sofa_cns'] = data['sofa_cns'].fillna(0)
data['sofa_renal'] = data['sofa_renal'].fillna(0)
data['sofa_resp'] = data['sofa_resp'].fillna(0)

# one-hot encoding of FiO2 -> this way we avoid having to impute FiO2 for vent_status = 1
data['FiO2'] = data['FiO2'].fillna(0) 
data['FiO2_0'] = data.apply(lambda row: 21          if row['ventilation_status'] == 0 else 0, axis=1)
data['FiO2_2'] = data.apply(lambda row: row['FiO2'] if row['ventilation_status'] == 2 else 0, axis=1)
data['FiO2_3'] = data.apply(lambda row: row['FiO2'] if row['ventilation_status'] == 3 else 0, axis=1)
data['FiO2_4'] = data.apply(lambda row: row['FiO2'] if row['ventilation_status'] == 4 else 0, axis=1)

# and same for these deltas
data['delta_FiO2'] = data['delta_FiO2'].fillna(0)
data['delta_FiO2_0'] = data.apply(lambda row: row['delta_FiO2'] if row['ventilation_status'] == 0 else 0, axis=1)
data['delta_FiO2_2'] = data.apply(lambda row: row['delta_FiO2'] if row['ventilation_status'] == 2 else 0, axis=1)
data['delta_FiO2_3'] = data.apply(lambda row: row['delta_FiO2'] if row['ventilation_status'] == 3 else 0, axis=1)
data['delta_FiO2_4'] = data.apply(lambda row: row['delta_FiO2'] if row['ventilation_status'] == 4 else 0, axis=1)

# Replace missing labs and vitals with their normal ranges
midpoint_map = {
    'mch': 29.8,
    'mchc': 33.6,
    'mcv': 90.0,
    'platelet': 300.0,
    'rbc': 5.0,
    'rdw': 13.0,
    'wbc': 7.25,
    'inr': 1.0,
    'pt': 10.75,
    'ptt': 28.5,
    'alt': 31.5,
    'alp': 95.5,
    'ast': 25.0,
    'bilirubin_total': 0.65,
    'albumin': 4.4,
    'aniongap': 10.0,
    'bicarbonate': 25.0,
    'bun': 13.5,
    'calcium': 9.5,
    'chloride': 101.0,
    'creatinine': 0.9,
    'glucose': 84.5,
    'sodium': 140.0,
    'potassium': 4.25,
    'ph': 7.4,
    'lactate': 1.05,
    'heart_rate': 80.0,
    'mbp': 87.5,
    'resp_rate': 16.0,
    'temperature': 36.65
}

data = data.fillna(midpoint_map)

# Hemoglobin & Hematocrit by hand because depends on sex
data.loc[data['gender'] == 'M', 'hemoglobin'] = data.loc[data['gender'] == 'M', 'hemoglobin'].fillna(15.5)
data.loc[data['gender'] == 'F', 'hemoglobin'] = data.loc[data['gender'] == 'F', 'hematocrit'].fillna(13.7)
data.loc[data['gender'] == 'M', 'hematocrit'] = data.loc[data['gender'] == 'M', 'hemoglobin'].fillna(44.4)
data.loc[data['gender'] == 'F', 'hematocrit'] = data.loc[data['gender'] == 'F', 'hematocrit'].fillna(39.7)

# Replace missing values in deltas with 0, after computing the inverse
deltas = ['delta_FiO2_0', 'delta_FiO2_2', 'delta_FiO2_3', 'delta_FiO2_4',
          'delta_sofa_coag', 'delta_sofa_liver', 'delta_sofa_cv',
          'delta_sofa_cns', 'delta_sofa_renal', 'delta_sofa_resp',
          'delta_hemoglobin', 'delta_hematocrit', 'delta_mch', 'delta_mchc',
          'delta_mcv', 'delta_platelet', 'delta_rbc', 'delta_rdw', 'delta_wbc',
          'delta_inr', 'delta_pt', 'delta_ptt', 'delta_alt', 'delta_alp',
          'delta_ast', 'delta_bilirubin_total',
          'delta_albumin', 'delta_aniongap', 'delta_bicarbonate', 'delta_bun', 'delta_calcium',
          'delta_chloride', 'delta_creatinine', 'delta_glucose_lab', 'delta_sodium',
          'delta_potassium', 'delta_ph', 'delta_lactate', 'delta_heart_rate', 'delta_mbp',
          'delta_resp_rate', 'delta_temperature', 'delta_glucose', 'delta_heart_rhythm']

# get the inverse for the deltas -> the higher the delta the less accurate the feature is
for d in deltas:
    data[d] = data[d].apply(lambda x: 1/x if x != 0 else 1)

# impute missing data with 0 for the deltas
data[deltas] = data[deltas].fillna(0)

# Verify that there are no more missing values that matter
null_cols = data.columns[data.isna().any()].tolist()
print(null_cols)

# Save data
data.to_csv('data/MIMIC_IV_clean.csv', index=False)