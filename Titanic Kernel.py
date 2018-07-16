import numpy as np
import pandas as pd
import os
print(os.listdir("../input"))
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import GridSearchCV
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

# Import Data and Set Up Train and Test Datasets
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
data = train.append(test)
data['FamilySize'] = data['SibSp'] + data['Parch']
train['FamilySize'] = data['FamilySize'][:891]
test['FamilySize'] = data['FamilySize'][891:]

# Extracting Titles to Fill in NaN Values for Age
data['Title'] = data['Name']
# Cleaning name and extracting Title
for name_string in data['Name']:
    data['Title'] = data['Name'].str.extract('([A-Za-z]+)\.', expand=True)

# Map Less Common Titles to More Common Ones
mapping = {'Miss' : 3, 'Mr': 4, 'Mrs': 5, 'Mlle': 3, 'Major': 4, 'Col': 4, 'Sir': 4, 'Don': 4, 'Mme': 3,
           'Jonkheer': 4, 'Lady': 5, 'Capt': 4, 'Countess': 5, 'Ms': 3, 'Dona': 5,
           'Dr' : 1, 'Master': 2, 'Rev' : 6}
data.replace({'Title': mapping}, inplace=True)
titles = [1, 2, 3, 4, 5, 6]

# For Missing Age Values Use Title to Find Mean Age of The Group
for bin in titles:
    age = data.groupby('Title')['Age'].median()[bin]
    data.loc[(data['Age'].isnull()) & (data['Title'] == bin), 'Age'] = age
del data['Title']

# Encode Sec
data['Sex'] = data['Sex'].map( {'female': 0, 'male': 1} )
train['Sex'] = data['Sex'][:891]
test['Sex'] = data['Sex'][891:]

# Use Mean Value to Fill in Missing Fare Information
data['Fare'] = data['Fare'].fillna(data['Fare'].mean())

# Now Group by Last Name and Fare (same fare is paid by a group
# that bought tickets together) to Determine Groups of Passengers
# And See if People in The Same Group Survive
#--------------- Credit to Shunjiang Xu who came up with the idea for this feature------------------------------
#--------------- And Konstantin Masich who cleaned this code-----------------------------------------------------
#   https://www.kaggle.com/shunjiangxu  https://www.kaggle.com/konstantinmasich
data['Last name'] = data['Name'].apply(lambda x: str.split(x, ",")[0])
Default = 0.5
data['Family_Survival'] = Default
for _, groupedDf in data.groupby(['Last name', 'Fare']):
    if (len(groupedDf) != 1):
    # A Family group is found.
        for i, bracket in groupedDf.iterrows():
            max_survived = groupedDf.drop(i)['Survived'].max()
            min_survived = groupedDf.drop(i)['Survived'].min()
            pID = bracket['PassengerId']
            if (max_survived == 1.0):
                data.loc[data['PassengerId'] == pID, 'Family_Survival'] = 1
            elif (min_survived == 0.0):
                data.loc[data['PassengerId'] == pID, 'Family_Survival'] = 0

for _, groupedDf in data.groupby('Ticket'):
    if (len(groupedDf) != 1):
        for i, row in groupedDf.iterrows():
            if (row['Family_Survival'] == 0) | (row['Family_Survival']== 0.5):
                smax = groupedDf.drop(i)['Survived'].max()
                smin = groupedDf.drop(i)['Survived'].min()
                passID = row['PassengerId']
                if (smax == 1.0):
                    data.loc[data['PassengerId'] == passID, 'Family_Survival'] = 1
                elif (smin==0.0):
                    data.loc[data['PassengerId'] == passID, 'Family_Survival'] = 0
#---------------------------------------------------------------------------------------------------------------

# Update columns in train and test from data and segment age and fare into bins
train['Family_Survival'] = data['Family_Survival'][:891]
test['Family_Survival'] = data['Family_Survival'][891:]
label = LabelEncoder()
data['AgeBin'] = pd.qcut(data['Age'], 5)
data['FareBin'] = pd.qcut(data['Fare'], 5)
data['Age'] = label.fit_transform(data['AgeBin'])
data['Fare'] = label.fit_transform(data['FareBin'])
del data['AgeBin']
del data['FareBin']
train['Age'] = data['Age'][:891]
test['Age'] = data['Age'][891:]
train['FamilySize'] = data['FamilySize'][:891]
test['FamilySize'] = data['FamilySize'][891:]
train['Fare'] = data['Fare'][:891]
test['Fare'] = data['Fare'][891:]
data.drop(['Name','Last name', 'Embarked', 'SibSp', 'Parch', 'Ticket', 'Cabin',], axis = 1, inplace = True)
test.drop(['Name', 'Embarked', 'SibSp', 'Parch', 'Ticket', 'Cabin',], axis = 1, inplace = True)
train.drop(['Name', 'Embarked', 'SibSp', 'Parch', 'Ticket', 'Cabin',], axis = 1, inplace = True)

# Scale features to be used in KNN
X  = train.drop('Survived', 1)
Y = train['Survived']
X_test = test.copy()
std_scaler = StandardScaler()
X = std_scaler.fit_transform(X)
X_test = std_scaler.transform(X_test)

# Acknowledgement: Tried using countless classifiers but could not get better results than with
# KNN, parameters taken from Konstantin Masich's kernel https://www.kaggle.com/konstantinmasich
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(algorithm='auto', leaf_size=26, metric='minkowski',
                            metric_params=None, n_jobs=1, n_neighbors=6, p=2,
                            weights='uniform')

# Fit model and predict results
knn.fit(X, Y)
prediction = knn.predict(X_test)
submission = pd.DataFrame(pd.read_csv("../input/test.csv")['PassengerId'])
submission['Survived'] = prediction
submission.to_csv("../working/submission.csv", index = False)