
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
file_path = "E:\\CodeAlpha\\creditscore\\Credit Score Classification Dataset.csv"  # Replace with your file path
data = pd.read_csv(file_path)
print(data.head())

print(data.describe())

# Separate features and target variable
X = data.drop(columns='Credit Score')
y = data['Credit Score']

# Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Preprocess categorical features using one-hot encoding
categorical_features = ['Gender', 'Education', 'Marital Status', 'Home Ownership']
numerical_features = ['Age', 'Income', 'Number of Children']

# OneHotEncoder will be applied to categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Create a pipeline with the preprocessor and classifier
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

# Print the results
print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(report)


# Function to get user input and predict credit score
def predict_credit_score():
    print("\nEnter the details to predict the credit score:")
    
    # User input
    age = int(input("Age: "))
    gender = input("Gender (Male/Female): ")
    income = int(input("Income: "))
    education = input("Education (High School Diploma/Bachelor's Degree/Master's Degree/Doctorate): ")
    marital_status = input("Marital Status (Single/Married): ")
    num_children = int(input("Number of Children: "))
    home_ownership = input("Home Ownership (Owned/Rented): ")
    
    # Create a DataFrame from user input
    user_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Income': [income],
        'Education': [education],
        'Marital Status': [marital_status],
        'Number of Children': [num_children],
        'Home Ownership': [home_ownership]
    })
    
    # Predict the credit score using the trained model
    prediction = model.predict(user_data)
    
    # Decode the prediction
    predicted_credit_score = label_encoder.inverse_transform(prediction)
    
    print(f"\nPredicted Credit Score: {predicted_credit_score[0]}")

# Call the function to predict credit score based on user input
predict_credit_score()

