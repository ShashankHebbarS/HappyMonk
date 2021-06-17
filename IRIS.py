import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, r2_score
%matplotlib inline

df = pd.read_csv(r'C:\Shashank\py\ml\iris.csv')
df.isnull().sum()
df.head()

plt.figure(figsize=(14,10))
plt.subplot(2, 2, 1)
sns.scatterplot(x=df.SepalLengthCm, y=df.SepalWidthCm, data=df, hue="Species")
plt.subplot(2, 2, 2)
sns.scatterplot(x=df.PetalLengthCm, y=df.PetalWidthCm, data=df, hue="Species")
plt.subplot(2, 2, 3)
sns.scatterplot(x=df.SepalLengthCm, y=df.PetalLengthCm, data=df, hue="Species")
plt.subplot(2, 2, 4)
sns.scatterplot(x=df.SepalWidthCm, y=df.PetalWidthCm, data=df, hue="Species")
plt.show()

df.drop(['Id'], axis=1, inplace=True)
df['Species'] = df['Species'].replace({'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2})

sns.heatmap(df.corr(), annot=True)
plt.show()

y = df[['Species']]
X = df.drop(['Species'], axis=1)

ss = StandardScaler()
ss.fit(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(10, input_dim=4, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

tmp = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=200)
print(tmp)

print(model.evaluate(X_test, y_test))

pred = model.predict(X_test)
print(pred)

#print(model.summary())

#tmp.history

d = {
    'Train Loss'     : tmp.history['loss'],
    'Train Accuracy' :  tmp.history['accuracy'],
    'Test Loss'      : tmp.history['val_loss'],
    'Test Accuracy'  : tmp.history['val_accuracy']
    }
data = pd.DataFrame(data=d)

sns.lineplot(x=data['Train Loss'], y=data['Test Loss'])
plt.show()

sns.lineplot(x=data['Train Accuracy'], y=data['Test Accuracy'])
plt.show()

epochs = range(1,201)
sns.lineplot(x=epochs, y=data['Train Loss'])
sns.lineplot(x=epochs, y=data['Test Loss'])
plt.xlabel('Epochs')
plt.show()

y_pred = [rows.argmax() for rows in pred]
    
#r2_score(y_test, pred)
t = { 'Test' : y_test.Species,
        'Pred' : y_pred   
        }
a = pd.DataFrame(data=t)
a.reset_index().drop(['index'], axis=1)

