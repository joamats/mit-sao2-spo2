from utils import *
import shap
from tabulate import tabulate
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TKAgg')

# Load test data
X_test = pd.read_csv('data/ML/X_test.csv')

# Load the model from a file
filename = "models/xgbr.pkl"
model = load_model(filename)

# Get Feature Importance from XGBoost
importances = model.feature_importances_

# Sort the feature importances in descending order
sorted_importances = sorted(zip(X_test.columns, importances),
                            key=lambda x: x[1], reverse=True)

# Create a list of the top 20 features and their importance scores
top_features = []
for i in range(20):
    top_features.append([i+1, sorted_importances[i][0], sorted_importances[i][1]])

# Print the table using tabulate
print(tabulate(top_features,
               headers=["Rank", "Feature", "Importance"],
               tablefmt='github'))

# calculate SHAP values for the test data
explainer = shap.Explainer(model)
shap_values = explainer(X_test)

# plot the SHAP values for each feature
fig, ax = plt.subplots(figsize=(6, 8))
shap.plots.beeswarm(shap_values, max_display=10, show=False)
plt.tight_layout()
plt.savefig('results/xgb_shap_plot.png', dpi=300)

# plot the most important features
fig, ax = plt.subplots(figsize=(6, 8))
shap.plots.bar(shap_values, max_display=10, show=False)
plt.tight_layout()
plt.savefig('results/xgb_ft_plot.png', dpi=300)