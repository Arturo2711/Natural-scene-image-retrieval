from sklearn.naive_bayes import GaussianNB
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Function to display confusion matrix
def display_matrix(cm, labels):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

# Function to display classification report and confusion matrix
def report(y, y_pred):
    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']  # Assuming 10 clusters based on your previous example
    cm = confusion_matrix(y, y_pred)
    classification_rep = classification_report(y, y_pred, labels=labels)
    print('Confusion Matrix:')
    print(cm)
    print('Classification Report:')
    print(classification_rep)
    display_matrix(cm, labels)

# Load the pre-clustered data
df = pd.read_csv('Classifier/describing_vectors_clustered.csv')

# Select feature columns and target
features = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']
X = df[features]
y = df['cluster']  # Assuming 'cluster' is the column containing cluster labels

# Initialize and train the Gaussian Naive Bayes model
model = GaussianNB()
model.fit(X, y)

# Predict using the trained model
y_pred = model.predict(X)

# Optional: Report classification metrics (you may want to split data for proper evaluation)
report(y, y_pred)

# Save the trained model
#joblib.dump(model, 'bayesian.pkl')
#print('Model has been saved successfully.')
