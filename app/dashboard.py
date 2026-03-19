import pandas as pd
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
import joblib

def main():
    X_test_transformed = pd.read_csv('X_test_transformed.csv')
    y_test_full = pd.read_csv('y_test_full.csv')['Rings']
    model = joblib.load("model.joblib")
    explainer = ClassifierExplainer(model, X_test_transformed, y_test_full)
    db = ExplainerDashboard(explainer)
    db.to_yaml("dashboard.yaml", explainerfile="explainer.joblib", dump_explainer=True)

if __name__ == "__main__":
    main()
