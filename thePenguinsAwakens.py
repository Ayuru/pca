import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA

input_path = Path(r"C:\Users\ayuru\Desktop\Py\penguins\penguins.csv")
penguins = pd.read_csv(input_path)

print(penguins.head())
print(penguins.info())
print(penguins.describe())
print(penguins.isnull().sum())
print(penguins['Species'].value_counts())

penguins = penguins.dropna()

sns.countplot(data = penguins, x = 'Species')
plt.title('Liczba obserwacji dla poszczególnych gatunków')
plt.show()

features = ['CulmenLength', 'CulmenDepth', 'FlipperLength', 'BodyMass']

for feature in features:
    sns.boxplot(data = penguins, x = 'Species', y = feature)
    plt.title(f'{feature} w zależności od gatunku')
    plt.show()

sns.pairplot(penguins, hue = 'Species')
plt.show()


plt.figure(figsize = (8, 6))
sns.heatmap(penguins.corr(), annot = True)
plt.title('Macierz korelacji')
plt.show()


X = penguins.drop('Species', axis = 1)
y = penguins['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0, stratify = y)


scaler = StandardScaler()

X_train_standardized = scaler.fit_transform(X_train)
X_test_standardized = scaler.transform(X_test)


logistic_regression = LogisticRegression(max_iter = 1000)

params_lr = {'C': [0.007, 0.02, 0.13, 1, 44, 666]}

lr_gridsearch = GridSearchCV(logistic_regression,
                             params_lr,
                             scoring = 'f1_macro',
                             cv = 5,
                             n_jobs = -1)

lr_gridsearch.fit(X_train_standardized, y_train)

print('\nBest hyperparameter:', lr_gridsearch.best_params_)

model_lr = lr_gridsearch.best_estimator_

y_train_pred = model_lr.predict(X_train_standardized)
y_test_pred = model_lr.predict(X_test_standardized)

print('Logistic Regression F1 train:', f1_score(y_train, y_train_pred, average = 'macro'))
print('Logistic Regression F1 test:', f1_score(y_test, y_test_pred, average = 'macro'))


model_knn = KNeighborsClassifier()

params_knn = {'n_neighbors': [3, 7, 13, 21, 29], 'metric': ['euclidean', 'manhattan']}

knn_gridsearch = GridSearchCV(model_knn,
                              params_knn,
                              scoring = 'f1_macro',
                              cv = 5,
                              n_jobs = -1)

knn_gridsearch.fit(X_train_standardized, y_train)

print('\nBest hyperparameter:', knn_gridsearch.best_params_)

model_knn_v2 = knn_gridsearch.best_estimator_

y_train_pred = model_knn_v2.predict(X_train_standardized)
y_test_pred = model_knn_v2.predict(X_test_standardized)

print('KNN F1 train:', f1_score(y_train, y_train_pred, average = 'macro'))
print('KNN F1 test:', f1_score(y_test, y_test_pred, average = 'macro'))


model_tree = DecisionTreeClassifier()

params_tree = {'max_depth': [3, 5, 10, 20], 'min_samples_leaf': [3, 5, 10, 15]}

tree_gridsearch = GridSearchCV(model_tree,
                               params_tree,
                               scoring = 'f1_macro',
                               cv = 5,
                               n_jobs = -1)

tree_gridsearch.fit(X_train, y_train)

print('\nBest hyperparameter:', tree_gridsearch.best_params_)

model_tree_v2 = tree_gridsearch.best_estimator_

y_train_pred = model_tree_v2.predict(X_train)
y_test_pred = model_tree_v2.predict(X_test)

print('Decision Tree F1 train:', f1_score(y_train, y_train_pred, average = 'macro'))
print('Decision Tree F1 test:', f1_score(y_test, y_test_pred, average = 'macro'))


model_svm = SVC()

params_svm = {'C': [0.007, 0.02, 0.13, 1, 44, 666], 'kernel': ['linear', 'rbf', 'poly']}

svm_gridsearch = GridSearchCV(model_svm,
                              params_svm,
                              scoring = 'f1_macro',
                              cv = 5,
                              n_jobs = -1)

svm_gridsearch.fit(X_train_standardized, y_train)

print('\nBest hyperparameter:', svm_gridsearch.best_params_)

model_svm_v2 = svm_gridsearch.best_estimator_

y_train_pred = model_svm_v2.predict(X_train_standardized)
y_test_pred = model_svm_v2.predict(X_test_standardized)

print('SVM F1 train:', f1_score(y_train, y_train_pred, average = 'macro'))
print('SVM F1 test:', f1_score(y_test, y_test_pred, average = 'macro'))


random_forest = RandomForestClassifier(n_estimators = 1000, n_jobs = -1)

params_rf = {'max_depth': [3, 5, 10, 20], 'min_samples_leaf': [3, 5, 10, 15]}

rf_gridsearch = GridSearchCV(random_forest,
                             params_rf,
                             scoring = 'f1_macro',
                             cv = 5,
                             n_jobs = -1)

rf_gridsearch.fit(X_train, y_train)

print('\nBest hyperparameter:', rf_gridsearch.best_params_)

rf_model_v2 = rf_gridsearch.best_estimator_

y_train_pred = rf_model_v2.predict(X_train)
y_test_pred = rf_model_v2.predict(X_test)

print('Random Forest F1 train:', f1_score(y_train, y_train_pred, average = 'macro'))
print('Random Forest F1 test:', f1_score(y_test, y_test_pred, average = 'macro'))


model_adaboost = AdaBoostClassifier(estimator = DecisionTreeClassifier(max_depth = 1), n_estimators = 50)

params_adaboost = {'n_estimators': [2, 13, 67, 101, 666]}

adaboost_gridsearch = GridSearchCV(model_adaboost,
                                   params_adaboost,
                                   scoring = 'f1_macro',
                                   cv = 5,
                                   n_jobs = -1)

adaboost_gridsearch.fit(X_train, y_train)

print('\nBest hyperparameter:', adaboost_gridsearch.best_params_)

model_adaboost_v2 = adaboost_gridsearch.best_estimator_

y_train_pred = model_adaboost_v2.predict(X_train)
y_test_pred = model_adaboost_v2.predict(X_test)

print('AdaBoost F1 train:', f1_score(y_train, y_train_pred, average = 'macro'))
print('AdaBoost F1 test:', f1_score(y_test, y_test_pred, average = 'macro'))

pca = PCA(random_state = 42)
X_train_pca = pca.fit_transform(X_train_standardized)
X_test_pca = pca.transform(X_test_standardized)

print('Explained variance', pca.explained_variance_ratio_)

pca_results = []

for n_components in [4, 3, 2, 1]:
    model_pca = LogisticRegression(C = lr_gridsearch.best_params_['C'], max_iter = 1000)
    model_pca.fit(X_train_pca[:, :n_components], y_train)
    y_train_pred_pca = model_pca.predict(X_train_pca[:, :n_components])
    y_test_pred_pca = model_pca.predict(X_test_pca[:, :n_components])
    pca_results.append([n_components, f1_score(y_train, y_train_pred_pca, average = 'macro'), f1_score(y_test, y_test_pred_pca, average = 'macro')])

pca_results = pd.DataFrame(pca_results, columns = ['Liczba komponentów', 'F1 train', 'F1 test'])
print('\nPCA results:')
print(pca_results)