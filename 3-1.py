import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Step 1: Generate random data
np.random.seed(42)
X = np.random.randint(0, 1001, 300).reshape(-1, 1)
y = np.where((X > 500) & (X < 800), 1, 0).ravel()

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train Logistic Regression and SVM models
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

svm_clf = SVC(kernel='rbf', probability=True)
svm_clf.fit(X_train, y_train)

# Step 4: Predict using both models
y_pred_log_reg = log_reg.predict(X_test)
y_pred_svm = svm_clf.predict(X_test)

# Step 5: Plot the true labels and predictions from both models
plt.figure(figsize=(12, 6))

# Scatter plot for Logistic Regression
plt.subplot(1, 2, 1)
plt.scatter(X, y, color='gray', alpha=0.5, label='True Labels')
plt.scatter(X_test, y_pred_log_reg, color='blue', marker='x', label='Logistic Regression Predictions')
x_vals = np.linspace(0, 1000, 1000).reshape(-1, 1)
y_prob_logreg = log_reg.predict_proba(x_vals)[:, 1]
plt.plot(x_vals, y_prob_logreg, color='blue', linestyle='--', label='Logistic Regression Decision Boundary')
plt.title('Logistic Regression: True Labels and Predictions')
plt.xlabel('X values')
plt.ylabel('Labels')
plt.legend()
plt.grid(True)

# Scatter plot for SVM
plt.subplot(1, 2, 2)
plt.scatter(X, y, color='gray', alpha=0.5, label='True Labels')
plt.scatter(X_test, y_pred_svm, color='green', marker='s', label='SVM Predictions')
y_prob_svm = svm_clf.predict_proba(x_vals)[:, 1]
plt.plot(x_vals, y_prob_svm, color='green', linestyle='--', label='SVM Decision Boundary')
plt.title('SVM: True Labels and Predictions')
plt.xlabel('X values')
plt.ylabel('Labels')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
