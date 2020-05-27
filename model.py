import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import category_encoders as ce
from joblib import dump

# create df
trainval = pd.read_csv('./lotr_with_name_info.csv')
drop_cols = ['Unnamed: 0', 'Unnamed: 0.1']
trainval = trainval.drop(columns=drop_cols, axis=1)

# drop null values
trainval.dropna(inplace=True)

# setting features and target
target = 'race'
train_features = trainval.drop(columns = target)


def wrangle(X):
    X = X.copy()
    return X


train_race, val_race = train_test_split(trainval, random_state=42)
train = wrangle(train_race)
val = wrangle(val_race)
X_train = train.drop(columns=target)
X_val = val.drop(columns=target)
y_train = train[target]
y_val = val[target]

numeric_features = train_features.select_dtypes(include='number').columns.tolist()
cardinality = train_features.select_dtypes(exclude='number').nunique()
categorical_features = cardinality[cardinality <= 2000].index.tolist()

features = numeric_features + categorical_features

# model
model = make_pipeline(
    ce.OneHotEncoder(use_cat_names=True, handle_unknown = 'ignore'),
    SimpleImputer(strategy='median'),
    RandomForestClassifier(random_state=0, n_jobs=-1)
)

# Fit on train, score on val
model.fit(X_train, y_train)
model.predict(X_val)

print(model)
dump(model, 'model.pkl')
