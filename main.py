import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib


df = pd.read_csv('mmu_pd_features.csv')


X = df[['avg_step_length', 'std_step_length', 'avg_arm_swing', 'stability_index']]
y = df['label']

custom_weights = {0: 1, 1: 15, 2: 10, 3: 20}


model = RandomForestClassifier(
    n_estimators=200, 
    class_weight=custom_weights, 
    random_state=42,
    max_depth=12  
)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print(f"Improved Accuracy Score: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))


joblib.dump(model, 'pd_bagging_model.pkl')
print("Model 'pd_bagging_model.pkl' saved successfully.")