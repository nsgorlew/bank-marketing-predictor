import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# dataset: https://archive.ics.uci.edu/ml/datasets/bank+marketing
df = pd.read_csv('bank-full.csv',sep = ';')

# need feature column names and values for frontend
col_names = []
for col in df.columns:
    if df[col].dtypes == 'object':
        col_names.append(col)
features = {}
for col in col_names:
    features[col] = df[col].unique()


# need to make string features into numerical features
for col in df:
    if df[col].dtypes == 'object':
        encoder = LabelEncoder()
        encoder.fit(df[col])
        encoder_values = encoder.transform(df[col])
        df[col] = encoder_values


# split data into features and labels
# dropping pdays and previous because they are unchanged throughout dataset
df1 = df.drop(['pdays','previous'],axis=1)
X = df1.iloc[:,0:14]
y = df1.iloc[:,14]


# split data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)

model = RandomForestClassifier(n_estimators = 150, max_depth = 6)

# train
model.fit(X_train,y_train)
# test
print(model.score(X_test,y_test))

#save file
model_file = 'rf-bank-trained.sav'
pickle.dump(model,open(model_file,'wb'))




