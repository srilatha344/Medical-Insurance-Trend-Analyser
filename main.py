from datapreprocessing import load_data, show_data_summary, visualize_data
from modelling import split_data, preprocess_data, train_linear_regression, train_random_forest, evaluate_model, plot_feature_importance, plot_actual_vs_predicted

df = load_data('insurance.csv')

show_data_summary(df)
visualize_data(df)

X_train, X_test, y_train, y_test = split_data(df)
preprocessor = preprocess_data(X_train)

print("\nPreprocessed feature names:")
print(preprocessor.fit(X_train).get_feature_names_out())

linreg_model = train_linear_regression(X_train, y_train, preprocessor)

print("\nLinear Regression Model Evaluation:")
y_pred_linreg = evaluate_model(linreg_model, X_test, y_test)

rf_model = train_random_forest(X_train, y_train, preprocessor)

print("\nRandom Forest Regression Model Evaluation:")
y_pred_rf = evaluate_model(rf_model, X_test, y_test)

plot_actual_vs_predicted(y_test, y_pred_rf)
plot_feature_importance(rf_model, preprocessor)
