import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

df = pd.read_csv('Social_Network_Ads.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train, y_train)

# Finding Best Model
parameters = [{'C': [0.25,0.5,0.75,1], 'kernel': ['linear']},
              {'C': [0.25,0.5,0.75,1], 'kernel': ['rbf'], 'gamma': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}]
grid = GridSearchCV(classifier, param_grid=parameters, scoring='accuracy', cv=10, n_jobs=-1)
grid.fit(X_train, y_train)
best_acc = grid.best_score_
best_params = grid.best_params_
print("Best Accuracy: ", best_acc)
print("Best Parameters: ", best_params)
