from utils import *
import shap
from tabulate import tabulate
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TKAgg')

# Load test data
X_test = pd.read_csv('data/ML/X_test.csv')

withSOFA = True

if withSOFA:
    model_name = "xgbr_wSOFA"
else:
    model_name = "xgbr_woSOFA"    
    X_test = X_test.drop(columns=['sofa_resp'])
    X_test = X_test.drop(columns=['delta_sofa_resp'])

# Load the model from a file
filename = f"models/{model_name}.pkl"
model = load_model(filename)

# calculate SHAP values for the test data
explainer = shap.Explainer(model)
shap_values = explainer(X_test)

# plot the SHAP values for each feature
fig, ax = plt.subplots(figsize=(6, 8))
shap.plots.beeswarm(shap_values, max_display=10, show=False)
plt.title("Feature Importance: SHAP Values for Top 10 Features")
plt.tight_layout()
plt.savefig(f'results/shap_{model_name}.png', dpi=300)


# # plot the most important features
# fig, ax = plt.subplots(figsize=(6, 8))
# shap.plots.bar(shap_values, max_display=10, show=False)
# plt.tight_layout()
# plt.savefig('results/xgb_ft_plot.png', dpi=300)