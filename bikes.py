import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from math import sqrt
from matplotlib.ticker import MaxNLocator
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder, PowerTransformer, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from IPython.display import display
from sklearn.decomposition import PCA

input_path = Path(r"C:\Users\ayuru\Desktop\Py\bikes\daily-bike-share.csv")
bike_data_raw = pd.read_csv(input_path)

bike_data = bike_data_raw.copy()
bike_data.drop(['instant', 'dteday', 'yr'], axis=1, inplace=True)

numeric_features = ['temp', 'atemp', 'hum', 'windspeed']
categorical_features = ['season','mnth','holiday','weekday','workingday','weathersit']
target = 'rentals'

bike_data['difference_temp'] = (bike_data['atemp'] - bike_data['temp'])/bike_data['temp']
bike_data.drop(['atemp'], axis=1, inplace=True)
numeric_features = ['temp', 'difference_temp', 'hum', 'windspeed']

print(f'Numeric features: {numeric_features}')
print(f'Categorical features: {categorical_features}')
print(f'Target: {target}')
display(bike_data)


X = bike_data[['temp']].copy()
y = bike_data[target].copy()

print('X:')
print(X.values[:3])
print('\ny:')
print(y.values[:3])


plt.scatter(X, y, alpha=0.35)
plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

print(f'X.shape: {X.shape}, y.shape {y.shape}')
print()
print(f'X_train.shape: {X_train.shape}, y_train.shape {y_train.shape}')
print(f'X_test.shape: {X_test.shape}, y_test.shape {y_test.shape}')


lr_model = LinearRegression() # inicjalizacja modelu
lr_model.fit(X_train, y_train) # trenowanie modelu

print(lr_model.intercept_)
print(lr_model.coef_)


X_linspace = np.linspace(0, 1, 100)
y_fitted = lr_model.intercept_ + lr_model.coef_ * X_linspace

# Rysowanie wykresu
plt.scatter(X_train, y_train, color='green', alpha=0.5, label='Zbiór treningowy')
plt.scatter(X_test, y_test, color='red', alpha=0.5, label='Zbiór testowy')
plt.plot(X_linspace, y_fitted, color='black', label='Prognoza')
plt.legend(loc='upper left')
plt.show()


X_new = [[0.3],
         [0.7]]

predicted_data = lr_model.predict(X_new)
print(predicted_data)


y_predict_train = lr_model.predict(X_train)
y_predict_test = lr_model.predict(X_test)

print('Predykcja:')
print(np.round(y_predict_test[:5]).astype(int))
print('Wartość prawdziwa:')
print(y_test.values[:5])


print(f'Train R^2: {r2_score(y_train, y_predict_train)}')


print(f'Train MAE: {mean_absolute_error(y_train, y_predict_train)}')
print(f'Test MAE: {mean_absolute_error(y_test, y_predict_test)}')


print(f'Train MAPE: {mean_absolute_percentage_error(y_train, y_predict_train)}')
print(f'Test MAPE: {mean_absolute_percentage_error(y_test, y_predict_test)}')


print(f'Train MSE: {mean_squared_error(y_train, y_predict_train)}')
print(f'Test MSE: {mean_squared_error(y_test, y_predict_test)}')


print(f'Train RMSE: {sqrt(mean_squared_error(y_train, y_predict_train))}')
print(f'Test RMSE: {sqrt(mean_squared_error(y_test, y_predict_test))}')


errors = y_predict_test - y_test

plt.scatter(x=y_test, y=errors, alpha=0.25)
plt.axhline(0, color="r", linestyle="--")
plt.xlabel('True Valuey_test')
plt.ylabel('Residual')
plt.title(f'Plot of residuals')
plt.show()


plt.hist(errors, bins=20)
plt.axvline(errors.mean(), color='k', linestyle='dashed', linewidth=1)
plt.title(f'Histogram of residuals, errors mean = {np.round(errors.mean(), 2)}')
plt.show()


# tworzymy puste listy gdzie umieszczane będą metryki oceniające
r_2_train_list = []
rmse_train_list = []
rmse_test_list = []

# Tworzymy X, który zostanie wykorzystany dla wizualizacji naszego modelu
X_linespace = np.arange(X.min().iloc[0], X.max().iloc[0], step=0.005).reshape(-1, 1)

for degree in [1, 2, 3, 5, 10, 20]:
    # Transformacja naszego X
    poly_transformer = PolynomialFeatures(degree=degree)
    X_train_transformed = poly_transformer.fit_transform(X_train)

    # Trenowanie naszego modelu
    polynomial_regression = LinearRegression()
    polynomial_regression.fit(X_train_transformed, y_train)

    plt.figure(figsize=(9, 3))

    # Wizualizacja wartości ze zbioru treningowego oraz testowego
    plt.scatter(X_train, y_train, color='green', alpha=0.5, label='Zbiór treningowy')
    plt.scatter(X_test, y_test, color='red', alpha=0.5, label='Zbiór testowy')

    ########## Wizualizacja wielomianu ##########
    y_fitted = polynomial_regression.predict(poly_transformer.transform(X_linespace))
    plt.plot(X_linespace, y_fitted, color='black', label='Prognoza')
    plt.title(f'Degree {degree}')
    plt.legend(loc='upper left')
    plt.show()

    # Predykcja modelu na zbiorze treningowym
    prediction_train = polynomial_regression.predict(X_train_transformed)

    # Predykcja modelu na zbiorze testowym - najpierw jednak należy przekształcić zbiór testowy
    X_test_transformed = poly_transformer.transform(X_test)
    prediction_test = polynomial_regression.predict(X_test_transformed)

    # Ocena modeli
    r_2_train_list.append(r2_score(y_train, prediction_train))
    rmse_train_list.append(sqrt(mean_squared_error(y_train, prediction_train)))
    rmse_test_list.append(sqrt(mean_squared_error(y_test, prediction_test)))


fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot()

ax.plot([1, 2, 3, 5, 10, 20], rmse_train_list, color='green', label='RMSE - Zbiór treningowy')
ax.plot([1, 2, 3, 5, 10, 20], rmse_test_list, color='red', label='RMSE - Zbiór testowy')
ax.legend(loc='center')

ax.xaxis.set_major_locator(MaxNLocator(integer=True)) # Ustawienie typu int dla wartości z osi X
ax.set_ylabel('RMSE', size=13)
ax.set_xlabel('Stopień wielomianu', size=15)

# Dodanie drugiej osi y dla R^2
ax2 = ax.twinx()
ax2.plot([1, 2, 3, 5, 10, 20], r_2_train_list, color='blue', label='R^2 - Zbiór treningowy')
ax2.legend(loc='center right')
ax2.set_ylabel('R^2', size=13)

plt.show()


cv = KFold(n_splits=5, shuffle=False)


# Tworzymy Pipeline - najpierw tworzymy wielomian, następnie uczymy model
polynomial_regression_pipeline = make_pipeline(PolynomialFeatures(), LinearRegression())

# Nasz model sprawdzi te hiperparametry
params = {'polynomialfeatures__degree': [1, 2, 3, 4, 5]}

# Inicjalizujemy Pipeline
polynomial_regression_gridsearch = GridSearchCV(polynomial_regression_pipeline,
                                                params,
                                                scoring='neg_mean_squared_error',
                                                cv=cv,
                                                n_jobs=-1)

# Uczymy Grid Search, podajemy X_train - Pipeline za nas zrobi wielomian :)
polynomial_regression_gridsearch.fit(X_train, y_train)

print("\nNajlepsze hiperparametry:", polynomial_regression_gridsearch.best_params_, "\n")

# Przekazujemy najlepszy estymator
polynomial_regression_model = polynomial_regression_gridsearch.best_estimator_

predictions = polynomial_regression_model.predict(X_test)

print(f'RMSE: {np.sqrt(mean_squared_error(y_test, predictions))}')


# Podział na zmienne objaśniające i zmienną objaśnianą
X = bike_data[numeric_features].copy()
y = bike_data[target].copy()

# Podział na zbiór treningowy i testowy (taki sam podział co poprzednio)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Uczymy model
polynomial_regression_pipeline = make_pipeline(PolynomialFeatures(), LinearRegression())

params = {'polynomialfeatures__degree': [1, 2, 3, 4, 5]}

polynomial_regression_gridsearch = GridSearchCV(polynomial_regression_pipeline,
                                                params,
                                                scoring='neg_mean_squared_error',
                                                cv=cv, 
                                                n_jobs=-1)

polynomial_regression_gridsearch.fit(X_train, y_train)

print("\nNajlepsze hiperparametry:", polynomial_regression_gridsearch.best_params_, "\n")

polynomial_regression_model = polynomial_regression_gridsearch.best_estimator_

predictions = polynomial_regression_model.predict(X_test)

print(f'RMSE: {np.sqrt(mean_squared_error(y_test, predictions))}')


# Uczymy model
polynomial_regression_pipeline = make_pipeline(PolynomialFeatures(), ElasticNet())

params = {'polynomialfeatures__degree': [1, 2, 3, 4, 5],
          'elasticnet__alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0],
          'elasticnet__l1_ratio': np.arange(0, 1.1, 0.1)}

polynomial_regression_gridsearch = GridSearchCV(polynomial_regression_pipeline,
                                                params,
                                                scoring='neg_mean_squared_error',
                                                cv=cv,
                                                n_jobs=-1)

polynomial_regression_gridsearch.fit(X_train, y_train)

print("\nNajlepsze hiperparametry:", polynomial_regression_gridsearch.best_params_, "\n")

polynomial_regression_model = polynomial_regression_gridsearch.best_estimator_

predictions = polynomial_regression_model.predict(X_test)

print(f'RMSE: {np.sqrt(mean_squared_error(y_test, predictions))}')


# Podział na zmienne objaśniające i zmienną objaśnianą
X = bike_data[numeric_features + categorical_features].copy()
y = bike_data[target].copy()

# Podział na zbiór treningowy i testowy (taki sam podział co poprzednio)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

numeric_transformer = Pipeline(steps=[
    ('logtransformer', PowerTransformer()),
    ('standardscaler', StandardScaler()),
    ('polynomialfeatures', PolynomialFeatures())])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Tworzenie końcowego Pipeline, który będziemy trenować
final_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('regressor', ElasticNet())])

params = {'preprocessor__num__polynomialfeatures__degree': [1, 2, 3, 4, 5],
          'regressor__alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0],
          'regressor__l1_ratio': np.arange(0, 1.1, 0.1)}

final_polynomial_regression_gridsearch = GridSearchCV(final_pipeline,
                                                      params,
                                                      scoring='neg_mean_squared_error',
                                                      cv=cv,
                                                      n_jobs=-1)

final_polynomial_regression_gridsearch.fit(X_train, y_train)

print("\nNajlepsze hiperparametry:", final_polynomial_regression_gridsearch.best_params_, "\n")

final_polynomial_regression_model = final_polynomial_regression_gridsearch.best_estimator_

predictions = final_polynomial_regression_model.predict(X_test)

print(f'RMSE: {np.sqrt(mean_squared_error(y_test, predictions))}')


predictions_train = final_polynomial_regression_model.predict(X_train)

plt.scatter(predictions, y_test, alpha = 0.25)

min_value = min(predictions.min(), y_test.min())
max_value = max(predictions.max(), y_test.max())

plt.plot([min_value, max_value], [min_value, max_value], color = 'r', linestyle = '--')
plt.xlabel('Prediction')
plt.ylabel('True Value')
plt.title('Prediction vs True Value')
plt.show()

errors_final = predictions - y_test

plt.scatter(x = y_test, y = errors_final, alpha = 0.25)
plt.axhline(0, color = "r", linestyle = "--")
plt.xlabel('True Valuey_test')
plt.ylabel('Residual')
plt.title('Plot of residuals')
plt.show()

plt.hist(errors_final, bins = 20)
plt.axvline(errors_final.mean(), color = 'k', linestyle = 'dashed', linewidth = 1)
plt.title(f'Histogram of residuals, errors mean = {np.round(errors_final.mean(), 2)}')
plt.show()

X_first = bike_data[['temp']].copy()
y_first = bike_data[target].copy()

X_train_first, X_test_first, y_train_first, y_test_first = train_test_split(
    X_first, y_first, test_size = 0.2, shuffle = False)

y_predict_train_first = lr_model.predict(X_train_first)
y_predict_test_first = lr_model.predict(X_test_first)

print('\nPierwszy model:')
print(f'Train R^2: {r2_score(y_train_first, y_predict_train_first)}')
print(f'Test R^2: {r2_score(y_test_first, y_predict_test_first)}')
print(f'Train MAE: {mean_absolute_error(y_train_first, y_predict_train_first)}')
print(f'Test MAE: {mean_absolute_error(y_test_first, y_predict_test_first)}')
print(f'Train MAPE: {mean_absolute_percentage_error(y_train_first, y_predict_train_first)}')
print(f'Test MAPE: {mean_absolute_percentage_error(y_test_first, y_predict_test_first)}')
print(f'Train MSE: {mean_squared_error(y_train_first, y_predict_train_first)}')
print(f'Test MSE: {mean_squared_error(y_test_first, y_predict_test_first)}')
print(f'Train RMSE: {np.sqrt(mean_squared_error(y_train_first, y_predict_train_first))}')
print(f'Test RMSE: {np.sqrt(mean_squared_error(y_test_first, y_predict_test_first))}')

print('\nOstatni model:')
print(f'Train R^2: {r2_score(y_train, predictions_train)}')
print(f'Test R^2: {r2_score(y_test, predictions)}')
print(f'Train MAE: {mean_absolute_error(y_train, predictions_train)}')
print(f'Test MAE: {mean_absolute_error(y_test, predictions)}')
print(f'Train MAPE: {mean_absolute_percentage_error(y_train, predictions_train)}')
print(f'Test MAPE: {mean_absolute_percentage_error(y_test, predictions)}')
print(f'Train MSE: {mean_squared_error(y_train, predictions_train)}')
print(f'Test MSE: {mean_squared_error(y_test, predictions)}')
print(f'Train RMSE: {np.sqrt(mean_squared_error(y_train, predictions_train))}')
print(f'Test RMSE: {np.sqrt(mean_squared_error(y_test, predictions))}')

print('\nWniosek:')
print('Ostatni model osiąga lepsze wyniki niż pierwszy. Zarówno wyższe wartości R^2 jak i niższe MAE, MAPE, MSE i RMSE wskazują, że dokładniej przewiduje opisywane zagadnienie - co ma sens, ponieważ model ten wykorzystuje więcej informacji.')

pca_preprocessor = Pipeline(steps=[('logtransformer', PowerTransformer()), ('standardscaler', StandardScaler())])

X_train_numeric_standardized = pca_preprocessor.fit_transform(X_train[numeric_features])
X_test_numeric_standardized = pca_preprocessor.transform(X_test[numeric_features])

pca = PCA(random_state = 42)
X_train_pca = pca.fit_transform(X_train_numeric_standardized)
X_test_pca = pca.transform(X_test_numeric_standardized)

print('\nExplained variance:', pca.explained_variance_ratio_)

numeric_transformer_pca = Pipeline(steps=[('logtransformer', PowerTransformer()), ('standardscaler', StandardScaler()), ('pca', PCA(random_state = 42)), ('polynomialfeatures', PolynomialFeatures())])

categorical_transformer_pca = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown = 'ignore'))])

preprocessor_pca = ColumnTransformer(transformers=[('num', numeric_transformer_pca, numeric_features), ('cat', categorical_transformer_pca, categorical_features)])

final_pipeline_pca = Pipeline(steps=[('preprocessor', preprocessor_pca), ('regressor', ElasticNet())])

pca_results = []

for n_components in [4, 3, 2, 1]:
    final_pipeline_pca.set_params(preprocessor__num__pca__n_components = n_components, preprocessor__num__polynomialfeatures__degree = final_polynomial_regression_gridsearch.best_params_['preprocessor__num__polynomialfeatures__degree'], regressor__alpha = final_polynomial_regression_gridsearch.best_params_['regressor__alpha'], regressor__l1_ratio = final_polynomial_regression_gridsearch.best_params_['regressor__l1_ratio'])
    final_pipeline_pca.fit(X_train, y_train)
    predictions_pca = final_pipeline_pca.predict(X_test)

    r2_pca = r2_score(y_test, predictions_pca)
    mae_pca = mean_absolute_error(y_test, predictions_pca)
    mape_pca = mean_absolute_percentage_error(y_test, predictions_pca)
    mse_pca = mean_squared_error(y_test, predictions_pca)
    rmse_pca = sqrt(mse_pca)

    pca_results.append([n_components, r2_pca, mae_pca, mape_pca, mse_pca, rmse_pca])

pca_results = pd.DataFrame(pca_results, columns = ['Liczba komponentów', 'R2', 'MAE', 'MAPE', 'MSE', 'RMSE'])

print('\nPCA results:')
print(pca_results)

print('Dodanie PCA nie pomogło. Przy 4PC wyniki niemal identyczne, przy 3 i mniej nastapiło znaczne pogorszenie.')