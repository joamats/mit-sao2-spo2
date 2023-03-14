from utils import *
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TKAgg')

# Load test data
X_test = pd.read_csv('data/ML/X_test.csv')

# Load the model from a file
filename = "models/xgboost.pkl"
model = load_model(filename)

# calculate SHAP values for the test data
explainer = shap.Explainer(model)
shap_values = explainer(X_test)

# plot the SHAP values for each feature
fig, ax = plt.subplots(figsize=(10, 8))
shap.plots.beeswarm(shap_values, max_display=20, show=False)
plt.tight_layout()
plt.savefig('results/xgb_shap_plot.png', dpi=300)

# plot the most important features
fig, ax = plt.subplots(figsize=(10, 8))
shap.plots.bar(shap_values, max_display=20, show=False)
plt.tight_layout()
plt.savefig('results/xgb_ft_plot.png', dpi=300)