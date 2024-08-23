import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
file_path = "E:\\CodeAlpha\\heart disease classification dataset.csv"
df = pd.read_csv(file_path)

print(df.head(10))

df = df.drop(columns=['Unnamed: 0'])

# Handling missing values 
df['trestbps'] = df['trestbps'].fillna(df['trestbps'].median())
df['chol'] = df['chol'].fillna(df['chol'].median())
df['thalach'] = df['thalach'].fillna(df['thalach'].median())

df['sex'] = df['sex'].map({'male': 1, 'female': 0})
df['target'] = df['target'].map({'yes': 1, 'no': 0})

#  features and target
X = df.drop(columns=['target'])
y = df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluating the model's performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", report)
print("Confusion Matrix:\n", conf_matrix)


import matplotlib.pyplot as plt
import seaborn as sns


# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()



# Get user input for prediction
print("\nEnter the following details for disease prediction:")

age = float(input("Age: "))
sex = input("Sex (male/female): ").strip().lower()
sex = 1 if sex == 'male' else 0

cp = float(input("Chest pain type (1-4): "))
trestbps = float(input("Resting blood pressure: "))
chol = float(input("Serum cholesterol (mg/dl): "))
fbs = float(input("Fasting blood sugar > 120 mg/dl (1 = true; 0 = false): "))
restecg = float(input("Resting electrocardiographic results (0-2): "))
thalach = float(input("Maximum heart rate achieved: "))
exang = float(input("Exercise induced angina (1 = yes; 0 = no): "))
oldpeak = float(input("ST depression induced by exercise relative to rest: "))
slope = float(input("Slope of the peak exercise ST segment (1-3): "))
ca = float(input("Number of major vessels (0-3): "))
thal = float(input("Thalassemia (3 = normal; 6 = fixed defect; 7 = reversible defect): "))

# Create a DataFrame with the input
user_data = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'cp': [cp],
    'trestbps': [trestbps],
    'chol': [chol],
    'fbs': [fbs],
    'restecg': [restecg],
    'thalach': [thalach],
    'exang': [exang],
    'oldpeak': [oldpeak],
    'slope': [slope],
    'ca': [ca],
    'thal': [thal]
})

# Scale the user input data
user_data_scaled = scaler.transform(user_data)

# Predict using the trained model
user_prediction = model.predict(user_data_scaled)

# Display the prediction result
if user_prediction[0] == 1:
    print("The model predicts that you have heart disease.")
else:
    print("The model predicts that you do not have heart disease.")