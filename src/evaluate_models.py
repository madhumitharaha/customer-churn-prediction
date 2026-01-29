import pickle
from sklearn.metrics import accuracy_score, classification_report
from src.train_models import X_test, y_test

log_model = pickle.load(open("model/logistic_model.pkl", "rb"))
rf_model = pickle.load(open("model/random_forest_model.pkl", "rb"))

for name, model in {
    "Logistic Regression": log_model,
    "Random Forest": rf_model
}.items():
    pred = model.predict(X_test)
    print(f"\n{name}")
    print("Accuracy:", accuracy_score(y_test, pred))
    print(classification_report(y_test, pred))
