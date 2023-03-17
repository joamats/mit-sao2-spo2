from utils import *
from custom_loss import *
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
import json

# Load data, already split 80-20
X_train = pd.read_csv('data/ML/X_train.csv')
y_train = pd.read_csv('data/ML/y_train.csv')[['SaO2']]

#model
#Create an XGBoost regressor
model = XGBRegressor(objective=assym_loss, random_state=42)

param_grid = {
    'colsample_bytree': [0.8, 0.9],
    'gamma': [0, 0.75, 1.5],
    'max_depth': [3, 5, 7],
    'n_estimators': [100, 500, 1000],
    'reg_alpha': [0, 0.75, 1.5],
    'reg_lambda': [0, 0.75, 1.5],
    'subsample': [0.8, 0.9]
}

grid_search = GridSearchCV(estimator=model,
                           param_grid=param_grid,
                           cv=4, n_jobs=-1,
                           verbose=10,
                           scoring='neg_mean_squared_error')

# fit the grid search object to the data
grid_search.fit(X_train, y_train)

# get the best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f"Best params: {best_params}")
print(f"Best score: {best_score}")

json.dump(best_params,
          open('models/best_params_xgbr.txt', 'w'))

