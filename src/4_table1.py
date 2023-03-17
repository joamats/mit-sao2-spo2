from tableone import TableOne
import pandas as pd

data = pd.read_csv('data/MIMIC_IV_clean.csv')

# Groupby Variable
data['race_white'] = data.race_group.apply(lambda x: "White" if x == 1 else "Non-White")
groupby = ['race_group']

# Continuous Variables
data['los_hosp_dead'] = data[data.mortality_in == 1].los_hospital
data['los_hosp_surv'] = data[data.mortality_in == 0].los_hospital

data['los_icu_dead'] = data[data.mortality_in == 1].los_icu
data['los_icu_surv'] = data[data.mortality_in == 0].los_icu

data['sao2-spo2'] = data['SaO2'] - data['SpO2']

data['no_pairs'] = data.groupby('subject_id')['subject_id'].transform('count')

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
data['vasopressors'] = data['vasopressors'].map({1: 'Received', 0: 'No'})
data['rrt'] = data['rrt'].map({1: 'Received', 0: 'No'})
data['invasive_vent'] = data['invasive_vent'].map({1: 'Received', 0: 'No'})
data['hidden_hypoxemia'] = data['hidden_hypoxemia'].map({1: 'Present', 0: 'No'})
data['mortality_in'] = data['mortality_in'].map({1: 'Died', 0: 'Survived'})
data['language'] = data['language'].map({1: 'Proficient', 0: 'Limited Proficiency'})
data['heart_rhythm'] = data['heart_rhythm'].map({0: 'Normal (or near normal) Rhythm',
                                                 1: 'Bradycardia with Pacer',
                                                 2: 'Atrial Dysrhythmias',
                                                 3: 'Bundle Branch Blocks',
                                                 4: 'Ventricular Dysrhythmias (severe)'})
                                                 
order_s = {"gender": ["Female", "Male"],
           "mortality_in": ["Died", "Survived"],
           "language": ["Limited Proficiency", "Proficient"]
          }

order_p = {"vasopressors": ["Received", "No"],
           "rrt": ["Received", "No"],
           "invasive_vent": ["Received", "No"],
           "hidden_hypoxemia": ["Present", "No"],
           "ventilation_status": ["None", "Supplemental Oxygen",
                                "Non-Invasive Vent. / HFNC", "Invasive Vent.", "Tracheostomy"]}

limit_s = {"gender": 1,
           "mortality_in": 1,
           "language": 1
           }

limit_p = {"vasopressors": 1,
           "rrt": 1,
           "invasive_vent": 1,
           "hidden_hypoxemia": 1
           }

labls_p = {'anchor_age': 'Age',
           'gender': 'Sex',
           'mortality_in': "In-Hospital Mortality",
           'language': "English Proficiency",
           'race_group': "Race"
           }

labls_s = {'vasopressors': 'Vasopressor(s)',
           'rrt': "RRT",
           'invasive_vent': "Invasive Ventilation",
           'hidden_hypoxemia': "Hidden Hypoxemia",
           'ventilation_status': "Type of Ventilation",
          }

categ_p = ['hidden_hypoxemia', 'ventilation_status', 'invasive_vent',
           'rrt', 'vasopressors', 'heart_rhythm']

categ_s = ['mortality_in','gender', 'language',]

nonnorm_s = ['anchor_age',
             'los_icu_dead', 'los_icu_surv',
             'CCI', 'SOFA_admission',
             'no_pairs']

nonnorm_p = ['SaO2', 'SpO2', 'sao2-spo2', 'delta_SpO2', 
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


data_s = data.groupby("subject_id").first().reset_index()

# Create a TableOne 
table1_s = TableOne(data_s, columns=categ_s+nonnorm_s,
                    rename=labls_s, limit=limit_s, order=order_s, 
                    groupby=groupby, categorical=categ_s, nonnormal=nonnorm_s,
                    missing=False, overall=False,
                    dip_test=True, normal_test=True, tukey_test=True, htest_name=True)

table1_s.to_latex('EDA/table1_subjects.tex')
table1_s.to_excel('EDA/table1_subjects.xlsx')

# Create a TableOne 
table1_p = TableOne(data, columns=categ_p+nonnorm_p,
                    rename=labls_p, limit=limit_p, order=order_p, 
                    groupby=groupby, categorical=categ_p, nonnormal=nonnorm_p,
                    overall=False,
                    dip_test=True, normal_test=True, tukey_test=True, htest_name=True)

table1_p.to_latex('EDA/table1_pairs.tex')
table1_p.to_excel('EDA/table1_pairs.xlsx')
