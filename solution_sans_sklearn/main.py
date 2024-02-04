import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)

df1 = pd.read_csv("train.csv")
df2 = pd.read_csv("test.csv")

# print(df1.columns)


# en fonction de la tranche d'age
# df1['Tranche_age'] =
limites_tranches = list(range(0, 91, 10))
etiquettes_tranches = [f'{i}-{i+9}' for i in limites_tranches[:-1]]
df1['TrancheAge'] = pd.cut(df1['Age'], bins=limites_tranches, labels=etiquettes_tranches, right=False)
pivot_age = pd.pivot_table(df1, values='Survived', index=['TrancheAge'],aggfunc='mean')
df1['TrancheAge'].fillna('20-29', inplace=True)
# print(pivot_age)

# en ct du sexe
pivot_sex = pd.pivot_table(df1, values='Survived', index=['Sex'],aggfunc='mean')
# print(pivot_sex)

# en fct de la classe
pivot_class = pd.pivot_table(df1, values='Survived', index=['Pclass'],aggfunc='mean')
# print(pivot_class)

# en fonction de la famille
pivot_sibling = pd.pivot_table(df1, values='Survived', index=['SibSp'],aggfunc='mean')
# print(pivot_sibling)

# en fonction le nombre d'enfants et parents
pivot_parch = pd.pivot_table(df1, values='Survived', index=['Parch'],aggfunc='mean')
# print(pivot_parch)

# en fonction du prix du ticket
limites_tranches = [0,10,80,520]
etiquettes_tranches = ['0-9','10-79','80-520']
df1['fare_tranche'] = pd.cut(df1['Fare'], bins=limites_tranches, labels=etiquettes_tranches, right=False)
pivot_fare = pd.pivot_table(df1, values='Survived', index=['fare_tranche'],aggfunc='mean')
df1['fare_tranche'].fillna('10-79', inplace=True)
# print(pivot_fare)


# en fonction de la porte d'embarquement
pivot_embarked = pd.pivot_table(df1, values='Survived', index=['Embarked'],aggfunc='mean')
# print(pivot_embarked)

# en fonction de la cabine
df1['Cabin_lettre'] = df1['Cabin'].str[0]
df1['Cabin_lettre'].replace(np.nan, 'unknown', inplace=True)
pivot_cabin = pd.pivot_table(df1, values='Survived', index=['Cabin_lettre'],aggfunc='mean')
# print(pivot_cabin)



df1['Proba'] = (df1['TrancheAge'].map(pivot_age['Survived']).astype(float)*2 +
                df1['Sex'].map(pivot_sex['Survived']).astype(float)*3+
                df1['Pclass'].map(pivot_class['Survived']).astype(float)*3+
                df1['SibSp'].map(pivot_sibling['Survived']).astype(float) +
                df1['Parch'].map(pivot_parch['Survived']).astype(float) +
                df1['fare_tranche'].map(pivot_fare['Survived']).astype(float)+
                df1['Cabin_lettre'].map(pivot_cabin['Survived']).astype(float)*2)/13

# remplit la colonne de 1 si p>0.45 sinon de 0.
df1['Estimation'] = np.where((df1['Proba']>0.45),1,0)

# print((df1['Estimation'] == df1['Survived']).mean())

#########################DF2###################
# en fonction de la tranche d'age
# df1['Tranche_age'] =
limites_tranches = list(range(0, 91, 10))
etiquettes_tranches = [f'{i}-{i+9}' for i in limites_tranches[:-1]]
df2['TrancheAge'] = pd.cut(df2['Age'], bins=limites_tranches, labels=etiquettes_tranches, right=False)
df2['TrancheAge'].fillna('20-29', inplace=True)


# en fonction du prix du ticket
limites_tranches = [0,10,80,520]
etiquettes_tranches = ['0-9','10-79','80-520']
df2['fare_tranche'] = pd.cut(df2['Fare'], bins=limites_tranches, labels=etiquettes_tranches, right=False)
df2['fare_tranche'].fillna('10-79', inplace=True)




# en fonction de la cabine
df2['Cabin_lettre'] = df2['Cabin'].str[0]
df2['Cabin_lettre'].replace(np.nan, 'unknown', inplace=True)




df2['Proba'] = (df2['TrancheAge'].map(pivot_age['Survived']).astype(float)*2 +
                df2['Sex'].map(pivot_sex['Survived']).astype(float)*3+
                df2['Pclass'].map(pivot_class['Survived']).astype(float)*3+
                df2['SibSp'].map(pivot_sibling['Survived']).astype(float) +
                df2['Parch'].map(pivot_parch['Survived']).astype(float) +
                df2['fare_tranche'].map(pivot_fare['Survived']).astype(float)+
                df2['Cabin_lettre'].map(pivot_cabin['Survived']).astype(float)*2)/13


df2['Estimation'] = np.where((df2['Proba']>0.45),1,0)

df3 = pd.DataFrame({'PassengerId': df2['PassengerId'],
                    'Survived': df2['Estimation']})
df3.reset_index(drop=True, inplace=True)

# print(df3)
#
# df3.to_csv("Submition.csv",index=False)











