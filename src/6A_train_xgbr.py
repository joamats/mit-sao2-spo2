from utils import *
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

# Load data, already split 80-20
X_train = pd.read_csv('data/ML/X_train.csv')
y_train = pd.read_csv('data/ML/y_train.csv')

# model
# Create an XGBoost regressor
model = XGBRegressor()
model.fit(X_train, y_train)

# Save the model to a file
filename = "models/xgboost.pkl"
save_model(model, filename)


#


