import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
data=pd.read_csv("c:\\Users\\Lenovo\\Download\\chitti titanic.csv")
# print(data)
data=data.head()
# print(data)
data=data.tail()
# print(data)
data=data.describe()
# print(data)
data=data.shape
# print(data)
data=data.sample(10)
# print(data)
data=data.info()
# print(data)
data=data.count()
# print(data)
data=data['Sex'].unique()
# print(data)
data=data['Name'].unique()
# print(data)
data=data['Cabin'].unique()
# print(data)
data=data["Embarked"].unique()
# print(data)
data=data['Age'].unique()
# print(data)
data=data['Parch'].unique()
# print(data)
data=data['Sex'].value_counts()
# print(data)
data=data[data['Age'].isnull()]
# print(data)
data=data.iloc[10]
# print(data)
data=data.iloc[0:10]
# print(data)
data=data.iloc[0:400]
# print(data)
data=data.iloc[400:]
# print(data)
data=len(data)
# print(data)
data=data.sample(n=10)
# print(data)
data=data.expanding()
# print(data)
data=data.sample(frac=0.5)
# print(data)
data=data.plot.hist()
plt.show()
# print(data)
data=data.plot.scatter(x='Sex',y='Age')
plt.show()
# print(data)
data=data.hist(bins=10,figsize=(9,7),grid=False);
plt.show()
# print(data)
data=data.plot()
plt.show()
# print(data)
data=data.groupby('Sex').Survived.mean().plot(kind='pie',normalize=False)
plt.show()
# print(data)
data=data.Survived.value_counts().plot(kind='Pie',normalize=True)
plt.show()
# print(data)
data=data.Sex.value_counts().plot(kind='bar')
plt.show()
# print(data)
data=data.dropna()
# print(data)
data=data.fillna('x')
# print(data)

sns.countplot(data['Survived'])
plt.show()

sns.countplot(data['Pclass'])
plt.show()

data['Pclass'].value_counts().plot(kind='Pie',autopct=' % 2f ')
plt.show()

plt.hist(data['Age'],bins=50)
plt.show()

sns.displot(data['Age'])
plt.show()

sns.boxplot(data['Age'])
plt.show()

data=data['Age'].min()
# print(data)

sns.heatmap(data.isnull())
plt.show()

sns.pairplot(data)
plt.show()

sns.countplot(data=data,x="Embarked")
plt.show()

data=data.corr()
# print(data)

data=data['Age'].hist(bins=70)
plt.show()
# print(data)

plt.figure(figsize=(7,7))
plt.xlabel("Passenger class")
plt.ylabel("Number of Passengers survived from these class")
plt.title("Class of passengers survived")
plt.bar(data['Pclass'].value_counts().keys(),data['Pclass'].value_counts())
plt.show()

plt.figure(figsize=(7,7))
plt.xlabel("Age groups")
plt.ylabel("Number of Passengers")
plt.title("Age group of Passengers")
plt.bar(data['Age'].value_counts().keys(),data['Age'].value_counts())
plt.show()


plt.figure(figsize=(7,7))
plt.xlabel("Age groups")
plt.ylabel("Number of Passengers survived")
plt.title("Age group of Passengers who survived")
plt.bar(data['Age'].value_counts().keys(),data['Age'].value_counts())
plt.show()