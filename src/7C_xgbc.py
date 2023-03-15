from utils import *
from tabulate import tabulate
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, roc_curve, \
                            roc_auc_score, balanced_accuracy_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TKAgg')


# Load data, already split 80-20
X_train = pd.read_csv('data/ML/X_train.csv')
y_train = pd.read_csv('data/ML/y_train.csv')['hidden_hypoxemia']
X_test = pd.read_csv('data/ML/X_test.csv')
y_test = pd.read_csv('data/ML/y_test.csv')['hidden_hypoxemia']

# model
# Create an XGBoost regressor
model = XGBClassifier()
model.fit(X_train, y_train, sample_weight=y_train * 99 + 1)

# Save the model to a file
filename = "models/xgbc.pkl"
save_model(model, filename)

# Make predictions on the test set
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Calculate the confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# Calculate sensitivity and specificity
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

# Calculate ROC curve and AUC score
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_pred)

# Calculate balanced accuracy
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

# Create a list of metrics and their corresponding values
metrics = [("Sensitivity", sensitivity),
           ("Specificity", specificity),
           ("ROC AUC", auc_score),
           ("Bal. Accuracy", balanced_accuracy)]

# Print results in a tabular format
print(tabulate(metrics, headers=["Metric", "Value"], 
               tablefmt="psql", floatfmt=".3f"))

# Plot ROC curve and Save

# Plot ROC curve
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (AUC = {:.2f})'.format(auc_score))
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")

# Save plot to file
plt.savefig('results/roc.png')



