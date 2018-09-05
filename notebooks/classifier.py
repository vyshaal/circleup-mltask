import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

excel_file = "../spreadsheets/Data science take home Datasets.xlsx"

user_df = pd.read_excel(excel_file, "user")
user_features_df = pd.read_excel(excel_file, "user_features")
model_test_file_df = pd.read_excel(excel_file, "model_test_file")

user_df.set_index('user_id', inplace=True)
user_features_df.set_index('user_id', inplace=True)
model_test_file_df.set_index('user_id', inplace=True)

dataset = user_features_df.copy()
dataset['response'] = dataset.index.map(lambda x: user_df.loc[x][0])

training_features, test_features, training_target, test_target, = \
    train_test_split(dataset.iloc[:, :-1], dataset.iloc[:, -1], test_size=1/3, stratify=dataset.iloc[:, -1])

pipe = Pipeline([('oversample', SMOTE()),
                 ('clf', RandomForestClassifier(n_jobs=-1))])

skf = StratifiedKFold()
param_grid = {'oversample__ratio': [0.25, 0.5, 1],
              'clf__max_depth': [3, 5],
              'clf__max_features': ['sqrt', 'log2'],
              'clf__n_estimators': [25, 50, 100]}

grid = GridSearchCV(pipe, param_grid, return_train_score=False,
                    n_jobs=-1, scoring='f1', cv=skf)

grid.fit(training_features, training_target)

print(grid.best_params_)
print(grid.score(test_features, test_target))
print(precision_score(test_target, grid.predict(test_features)))
print(recall_score(test_target, grid.predict(test_features)))
print(f1_score(test_target, grid.predict(test_features)))
print(confusion_matrix(test_target, grid.predict(test_features)))

model_test_file_df['prediction'] = grid.predict(model_test_file_df)

print('END')