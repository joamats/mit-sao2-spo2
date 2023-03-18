from utils import *
from custom_loss import *
from xgboost import XGBRegressor
import json

# Load data, already split 80-20
X_train = pd.read_csv('data/ML/X_train.csv')
y_train = pd.read_csv('data/ML/y_train.csv')[['SaO2']]

X_test = pd.read_csv('data/ML/X_test.csv')
y_test = pd.read_csv('data/ML/y_test.csv')[['SaO2']]

# weights = pd.read_csv('data/ML/y_train.csv')[['hidden_hypoxemia']] * .9 + .1 # HH count 10x more

withSOFA = False

if withSOFA:
    model_name = "xgbr_wSOFA"
else:
    model_name = "xgbr_woSOFA"
    X_train = X_train.drop(columns=['sofa_resp'])
    X_train = X_train.drop(columns=['delta_sofa_resp'])
    X_test = X_test.drop(columns=['sofa_resp'])
    X_test = X_test.drop(columns=['delta_sofa_resp'])

# fit the XGBoost regressor with the best hyperparameters - loaded afer 7A1
best_params = {"objective": assym_loss,
               "colsample_bytree": 0.9,
               "gamma": 1.5,
               "max_depth": 3,
               "n_estimators": 100,
               "reg_alpha": 1.5,
               "reg_lambda": 1.5,
               "subsample": 0.9}

model = XGBRegressor(**best_params)
model.fit(X_train, y_train)

# Make predictions on the train set
evaluate_model(model, X_train, y_train, f"{model_name}_train")

# Save the model to a file
filename = f"models/{model_name}.pkl"
save_model(model, filename)

y_pred = model.predict(X_test) # maybe round down the values?

# Calculate the R2 score
print(f"R\u00B2 = {r2_score(y_test, y_pred):.3f}")

# Make predictions on the validation set
evaluate_model(model, X_test, y_test, model_name)

# sao2_spo2_plot(pd.DataFrame({'SaO2': y_test.SaO2,'SpO2': y_pred}),
#                'results/SaO2vsSpO2_xgbr', 'Corrected')
# sao2_spo2_plot(pd.DataFrame({'SaO2': y_test.SaO2,'SpO2': X_test.SpO2}),
#                'results/SaO2vsSpO2_base', 'Measured')