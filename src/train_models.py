import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from data_loader import load_data
from preprocessing import preprocess_data

df = load_data("data/raw/churn.csv")
df = preprocess_data(df)

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

log_model = LogisticRegression()
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)

log_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

pickle.dump(log_model, open("model/logistic_model.pkl", "wb"))
pickle.dump(rf_model, open("model/random_forest_model.pkl", "wb"))
pickle.dump(scaler, open("model/scaler.pkl", "wb"))
pickle.dump(X.columns, open("model/columns.pkl", "wb"))

print("✅ Both models trained and saved")
