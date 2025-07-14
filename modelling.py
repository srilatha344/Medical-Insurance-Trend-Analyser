from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def split_data(df):
    X = df.drop(columns='charges')
    y = df['charges']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def preprocess_data(X_train):
    categorical_features = ['sex', 'smoker', 'region']
    numerical_features = ['age', 'bmi', 'children']

    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return preprocessor

def train_linear_regression(X_train, y_train, preprocessor):
    linreg_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    linreg_model.fit(X_train, y_train)
    return linreg_model

def train_random_forest(X_train, y_train, preprocessor):
    rf_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    rf_model.fit(X_train, y_train)
    return rf_model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
    print("R-squared (R^2) Score:", r2_score(y_test, y_pred))
    return y_pred

def plot_feature_importance(rf_model, preprocessor):
    importances = rf_model.named_steps['regressor'].feature_importances_
    feature_names = preprocessor.get_feature_names_out()
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title('Feature Importance (Random Forest)')
    plt.show()

def plot_actual_vs_predicted(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([0, max(y_test)], [0, max(y_test)], color='red', linestyle='--')
    plt.xlabel('Actual Charges')
    plt.ylabel('Predicted Charges')
    plt.title('Actual vs Predicted Charges')
    plt.show()
