# Load the necessary libraries 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC 

# Load the iris dataset 
df = pd.read_csv(r"C:Iris.csv") 
print(df.columns)

# Split the data into features and labels
X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df['Species']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an SVM model and train it
model = SVC()
model.fit(X_train, y_train)

# Evaluate the model on the test data
accuracy = model.score(X_test, y_test)

print('Test accuracy:', accuracy)
