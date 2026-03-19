import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib

def main():
    # Получение данных
    df = pd.read_csv('abalone.csv')
    X_full = df.drop('Rings', axis=1)
    y_class = (df['Rings'] >= 10).astype(int)
    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X_full, y_class, test_size=0.25,
                                                                            random_state=42)
    # Подготовка данных для модели
    numerical_features = ['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight',
                          'Shell weight']
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaling', MinMaxScaler())
    ])
    ct = ColumnTransformer([
        ('ohe', OneHotEncoder(drop='first', handle_unknown="ignore"), ['Sex']),
        ('num', num_pipe, numerical_features)
    ])
    X_train_transformed = ct.fit_transform(X_train_full)
    X_test_transformed = ct.transform(X_test_full)
    new_features = list(ct.named_transformers_['ohe'].get_feature_names_out())
    new_features.extend(numerical_features)
    X_train_transformed = pd.DataFrame(X_train_transformed, columns=new_features)
    X_test_transformed = pd.DataFrame(X_test_transformed, columns=new_features)

    # Обучение модели
    model = RandomForestClassifier(random_state=42, n_jobs=-1)
    params = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    gs = GridSearchCV(model, params, scoring='accuracy', cv=3, n_jobs=-1, verbose=2)
    gs.fit(X_train_transformed, y_train_full)
    print(f"Лучшие параметры: {gs.best_params_}")
    print(f"Лучшая точность: {gs.best_score_:.4f}")
    model = gs.best_estimator_

    # Сохранение данных
    joblib.dump(model, 'model.joblib')
    X_test_transformed.to_csv('X_test_transformed.csv', index=False)
    y_test_full.to_csv('y_test_full.csv', index=False)

if __name__ == "__main__":
    main()
