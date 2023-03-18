import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit

# load data
data = pd.read_csv('data/MIMIC_IV_clean.csv')

bas = ['SpO2', 'delta_SpO2',
       'anchor_age','sex_female', 'race_group', 'language',
       'CCI', 'SOFA_admission']

rxs = ['ventilation_status', 'invasive_vent', 'rrt',
       'delta_vent_start', 'delta_rrt', 'delta_vp_start',
       'vasopressors', 'norepinephrine_equivalent_dose']

fts = ['FiO2_0', 'FiO2_2', 'FiO2_3', 'FiO2_4',      
       'sofa_coag', 'sofa_liver', 'sofa_cv', 'sofa_cns', 'sofa_renal', 'sofa_resp',
       'hemoglobin', 'hematocrit', 'mch', 'mchc', 'mcv', 'platelet',
       'rbc', 'rdw', 'wbc', 'inr', 'pt', 'ptt', 'alt', 'alp', 'ast',
       'bilirubin_total', 'albumin', 'aniongap',
       'bicarbonate', 'bun', 'calcium', 'chloride', 'creatinine',
       'glucose_lab', 'sodium', 'potassium', 'ph', 'lactate',
       'heart_rate', 'mbp', 'resp_rate', 'temperature',
       'glucose', 'heart_rhythm']

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
  
target = ['SaO2', 'hidden_hypoxemia'] 

# define X and y
X = data[bas + rxs + fts + deltas]
y = data[target]

# Initialize the GroupShuffleSplit object
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
# ss = ShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

groups = data.subject_id

train_idx, test_idx = next(gss.split(X, y, groups))
# train_idx, test_idx = next(ss.split(X, y))

# train val test split
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# Fill missing values with mean for now
X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_test.mean())

# Check dimensions
print("Train set shape: ", X_train.shape)
print("Test set shape: ", X_test.shape)

# Get the unique subject IDs in the training and testing sets
train_subjects = set(groups.iloc[train_idx])
test_subjects = set(groups.iloc[test_idx])

# Check if there are any subjects in both sets
if train_subjects.intersection(test_subjects):
    print("Error: Subjects in both training and testing sets")
else:
    print("Subjects in training and testing sets are unique")

# Save data, ML ready
X_train.to_csv('data/ML/X_train.csv', index=False)
y_train.to_csv('data/ML/y_train.csv', index=False)
X_test.to_csv('data/ML/X_test.csv', index=False)
y_test.to_csv('data/ML/y_test.csv', index=False)