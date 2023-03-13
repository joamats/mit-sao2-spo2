from tableone import TableOne
import pandas as pd

data = pd.read_csv('data/MIMIC_IV_clean.csv')

# Groupby Variable
data['race_white'] = data.race_group.apply(lambda x: "White" if x == 1 else "Non-White")
groupby = ['race_white']

# Continuous Variables
data['los_hosp_dead'] = data[data.mortality_in == 1].los_hospital
data['los_hosp_surv'] = data[data.mortality_in == 0].los_hospital

data['los_icu_dead'] = data[data.mortality_in == 1].los_icu
data['los_icu_surv'] = data[data.mortality_in == 0].los_icu

data['sao2-spo2'] = data['SaO2'] - data['SpO2']

nonnormal = ['SaO2', 'SpO2', 'delta_SpO2', 'sao2-spo2', 'anchor_age',
             'los_hosp_dead', 'los_hosp_surv',
             'los_icu_dead', 'los_icu_surv',
             'CCI', 'SOFA_admission',
             'sofa_coag', 'sofa_liver', 'sofa_cv',
             'sofa_cns', 'sofa_renal', 'sofa_resp','FiO2',
             'delta_vent_start', 'delta_rrt', 'delta_vp_start',
             'norepinephrine_equivalent_dose', 
             'hemoglobin', 'hematocrit', 'mch', 'mchc', 'mcv', 'platelet',
             'rbc', 'rdw', 'wbc', 
             'inr', 'pt', 'ptt', 'alt', 'alp', 'ast',
             'bilirubin_total', 
             'albumin', 'aniongap',
             'bicarbonate', 'bun', 'calcium', 'chloride', 'creatinine',
             'glucose_lab', 'sodium', 'potassium', 'ph', 'lactate',
             'heart_rate', 'mbp', 'resp_rate', 'temperature', 'glucose']

# Categorical Variables
data['vasopressors'] = data.norepinephrine_equivalent_dose.apply(lambda x: "Received" if x > 0 else "No")

data['race_group'] = data['race_group'].map({1: 'White',
                                             2: 'Hispanic',
                                             3: 'Asian',
                                             4: 'Other',
                                             5: 'Black'})

data['ventilation_status'] = data['ventilation_status'].map({0: 'None',
                                                             1: 'Supplemental Oxygen',
                                                             2: 'Non-Invasive Vent. / HFNC',
                                                             3: 'Invasive Vent.',
                                                             4: 'Tracheostomy'})

data['gender'] = data['gender'].map({'F': 'Female', 'M': 'Male'})
data['rrt'] = data['rrt'].map({1: 'Received', 0: 'No'})
data['invasive_vent'] = data['invasive_vent'].map({1: 'Received', 0: 'No'})
data['hidden_hypoxemia'] = data['hidden_hypoxemia'].map({1: 'Present', 0: 'No'})
data['mortality_in'] = data['mortality_in'].map({1: 'Died', 0: 'Survived'})
data['language'] = data['language'].map({1: 'Proficient', 0: 'Limited Proficiency'})

order = {"gender": ["Female", "Male"],
         "vasopressors": ["Received", "No"],
         "rrt": ["Received", "No"],
         "invasive_vent": ["Received", "No"],
         "hidden_hypoxemia": ["Present", "No"],
         "mortality_in": ["Died", "Survived"],
         "language": ["Limited Proficiency", "Proficient"], 
         "ventilation_status": ["None", "Supplemental Oxygen",
                                "Non-Invasive Vent. / HFNC", "Invasive Vent.", "Tracheostomy"]}

limit = {"gender": 1,
         "vasopressors": 1,
         "rrt": 1,
         "invasive_vent": 1,
         "hidden_hypoxemia": 1,
         "mortality_in": 1,
         "language": 1}

labels = {'gender': 'Sex',
          'vasopressors': 'Vasopressor(s)',
          'rrt': "RRT",
          'invasive_vent': "Invasive Ventilation",
          'hidden_hypoxemia': "Hidden Hypoxemia",
          'mortality_in': "In-Hospital Mortality",
          'language': "English Proficiency",
          'ventilation_status': "Type of Ventilation",
          'race_group': "Race"}

categorical = ['mortality_in','hidden_hypoxemia', 'gender', 'race_group', 'language',
               'ventilation_status', 'invasive_vent', 'rrt', 'vasopressors']

# Create a TableOne 
table1_raw = TableOne(data, columns=categorical+nonnormal,
                      rename=labels, order=order, limit=limit,
                      groupby=groupby, categorical=categorical, nonnormal=nonnormal,
                      smd=True,
                      dip_test=True, normal_test=True, tukey_test=True, htest_name=True)

table1_raw.to_excel('EDA/clean_table1.xlsx')