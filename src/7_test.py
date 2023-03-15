from utils import *
import tensorflow as tf

# Load test data
X_test = pd.read_csv('data/ML/X_test.csv')
y_test = pd.read_csv('data/ML/y_test.csv')

# Load the model from a file
filename = "models/multi"
model = tf.keras.models.load_model(filename)

y_pred = model.predict(X_test)[:,0]

print(f"R\u00B2 = {r2_score(y_test, y_pred):2f}")

# Make predictions on the validation set
evaluate_model(model, X_test, y_test, "multi")

