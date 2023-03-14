from utils import *

# Load test data
X_test = pd.read_csv('data/ML/X_test.csv')
y_test = pd.read_csv('data/ML/y_test.csv')

# Load the model from a file
filename = "models/xgboost.pkl"
model = load_model(filename)

y_pred = model.predict(X_test)

print(f"R2: {r2_score(y_test, y_pred)}")

# Make predictions on the validation set
#evaluate_model(model, X_test, y_test)

