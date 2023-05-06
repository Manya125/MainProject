import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split

from tensorflow.keras import Sequential, layers


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

pd.set_option('display.max_columns', None)
df = pd.read_csv('waterQuality1.csv')


df['is_safe'].unique()
rows_to_remove = list(df[df['is_safe'] == '#NUM!'].index)  # remove entries with #NUM

df.drop(rows_to_remove, inplace=True)
df['is_safe'] = df['is_safe'].astype('int64')  # converting object into int
df['ammonia'] = df['ammonia'].astype('float64')

X = df.drop(columns=['is_safe'])
y = df['is_safe']

sm = SMOTE(random_state=42)  # oversampling with smote
X, y = sm.fit_resample(X, y)


scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# keras sequential model
model = Sequential([
    layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
    layers.Dropout(0.2),

    layers.Dense(64, activation='relu'),
    layers.Dropout(0.1),

    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, validation_split=0.05)

predictions = model.predict(X_test)  # making prediction
y_pred = [1 if i > 0.5 else 0 for i in predictions]

pickle.dump(model,open('mdl.pkl','wb'))

accuracy_score(y_test, y_pred)  # evaluate the model

confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))

