import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load data (use train.csv for training and evaluation)
data = pd.read_csv("D:/web Project/PYTHON/Titanic-Survival-Prediction/data/tested.csv")

# Drop columns with too many missing values or irrelevant ones
data = data.drop(['Cabin', 'Name', 'Ticket'], axis=1)

# Fill missing values
data['Age'] = data['Age'].fillna(data['Age'].median())
data['Fare'] = data['Fare'].fillna(data['Fare'].median())
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

# Convert categorical to numeric
label_sex = LabelEncoder()
label_embarked = LabelEncoder()
data['Sex'] = label_sex.fit_transform(data['Sex'])
data['Embarked'] = label_embarked.fit_transform(data['Embarked'])

# Select features and target
X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = data['Survived']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Report:\n", classification_report(y_test, y_pred))
