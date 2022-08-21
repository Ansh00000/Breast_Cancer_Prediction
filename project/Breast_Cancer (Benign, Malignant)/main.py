# Importing Librarys.
from res.imports import *    # user defined library.

# Importing Dataset.
dataset = pd.read_csv("breast-cancer.csv")

# clearing Null values from dataset.
result = dataset.isin(["?"])==True
seriesobj = result.any()
columnname = list(seriesobj[seriesobj == True].index)
x = []
for col in columnname:
    rows = list(result[col][result[col]==True].index)
    for row in rows:
        x.append(row)
dataset = dataset.drop(x)

x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Splitting Dataset into Training and Test Set.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.2)

# Training Logistic Regression on Training Set.
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train, y_train)

# Predicting Test Set Value.
y_pred = classifier.predict(x_test)

# Making Confusion Matrix.
from sklearn.metrics import confusion_matrix    #       2(benign)                      4(malignant)
cm = confusion_matrix(y_test, y_pred)
print(cm)                                      # [[no. of correct predictions      [no. of incorrect prediction
                                               #    that tumour is benign ]     ,    that tumour is benign ]
                                               #  [no. of incorrect prediction     [ no. of correct predictions
                                               #  that tumour is malignant ]    ,    that tumour is malignant ]]

# Computing Accuracy with K-fold Cross Validation.
from sklearn.model_selection import cross_val_score
accuracy = cross_val_score(estimator=classifier, X=x_train, y=y_train, cv=10)
print("Accuracy : {: .2f} %".format(accuracy.mean() * 100))
print("Standard Deviation : {: .2f} %".format(accuracy.std() * 100))

