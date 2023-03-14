import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# load data
data = pd.read_csv('data/MIMIC_IV_clean.csv')

# impute missing data before split -> need to find a better way!
data.delta_vent_start = data.delta_vent_start.fillna(0)
data.delta_rrt = data.delta_rrt.fillna(0)
data.delta_vp_start = data.delta_vp_start.fillna(0)

fts = ['SpO2', 'delta_SpO2',
       'anchor_age','sex_female', 'race_group', 'language',
       'CCI', 'SOFA_admission',
       'sofa_coag', 'sofa_liver', 'sofa_cv',
       'sofa_cns', 'sofa_renal', 'sofa_resp','FiO2',
       'ventilation_status', 'invasive_vent', 'rrt', 'vasopressors',
       'delta_vent_start', 'delta_rrt', 'delta_vp_start',
       'norepinephrine_equivalent_dose', 
       'hemoglobin', 'hematocrit', 'mch', 'mchc', 'mcv', 'platelet',
       'rbc', 'rdw', 'wbc', 'inr', 'pt', 'ptt', 'alt', 'alp', 'ast',
       'bilirubin_total', 'albumin', 'aniongap',
       'bicarbonate', 'bun', 'calcium', 'chloride', 'creatinine',
       'glucose_lab', 'sodium', 'potassium', 'ph', 'lactate',
       'heart_rate', 'mbp', 'resp_rate', 'temperature', 'glucose'
      ]

target = ['SaO2'] 

# define X and y
X = data[fts]
y = data[target]

# Initialize the GroupShuffleSplit object
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

groups = data.subject_id

train_idx, test_idx = next(gss.split(X, y, groups))

# train val test split
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

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

# impute missing data after split -> check median vs mean
X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_test.mean())

# Save data, ML ready
X_train.to_csv('data/ML/X_train.csv', index=False)
y_train.to_csv('data/ML/y_train.csv', index=False)
X_test.to_csv('data/ML/X_test.csv', index=False)
y_test.to_csv('data/ML/y_test.csv', index=False)