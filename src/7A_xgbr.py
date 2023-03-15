from utils import *
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

# Load data, already split 80-20
X_train = pd.read_csv('data/ML/X_train.csv')
y_train = pd.read_csv('data/ML/y_train.csv')['SaO2']

# model
# Create an XGBoost regressor
model = XGBRegressor()
model.fit(X_train, y_train)

# Save the model to a file
filename = "models/xgboost.pkl"
save_model(model, filename)

# Load test data
X_test = pd.read_csv('data/ML/X_test.csv')
y_test = pd.read_csv('data/ML/y_test.csv')

y_pred = model.predict(X_test)

# Calculate the R2 score
print(f"R\u00B2 = {r2_score(y_test, y_pred):2f}")

# Make predictions on the validation set
evaluate_model(model, X_test, y_test, "xgb_results")