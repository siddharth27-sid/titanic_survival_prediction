import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
#load the dataset
df=pd.read_csv("Titanic-Dataset.csv")
#adding new features
df["Family"]=df["SibSp"]+df["Parch"]+1
df["Alone"]=df["Family"]==1
#mapping categorical features to numerical values
df["Alone"]=df["Alone"].map({True: 1, False: 0})
df["Embarked"]=df["Embarked"].map({"C":0,"Q":1,"S":2})
df["Sex"]=df["Sex"].map({"male":0,"female":1})
#data cleaning 
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
#loadind data to the model
x=df[["Sex","Age","Fare","Embarked","Alone","Family","Pclass"]]
y=df["Survived"]
#splitting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2,random_state=27)
#training the model
model3 = RandomForestClassifier(random_state=27)
model3.fit(x_train, y_train)
#saving the model
with open("titanic_model.pkl", "wb") as file:
    pickle.dump(model3, file)
print("Model trained and saved successfully!")
