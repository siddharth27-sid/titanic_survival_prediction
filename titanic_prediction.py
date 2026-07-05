import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
df=pd.read_csv("Titanic-Dataset.csv")
print (df.columns)
print (df.shape)
print(df.info())
print(df.isnull().sum())
print(df.head())
print(df["Embarked"].unique())
print(df["Survived"].value_counts(1)*100)
print(df["Survived"].mean() * 100)
print(pd.crosstab(df["Sex"], df["Survived"]))
print(pd.crosstab( df["Sex"],df["Survived"],normalize="index")* 100)
df["Embarked"].value_counts().plot(kind="pie",autopct="%1.1f%%")
plt.show(block=False)
plt.pause(3)
plt.close()
df["Age"] = df["Age"].fillna(df["Age"].median())
print(df.isnull().sum())
df["Family"]=df["SibSp"]+df["Parch"]+1
df["Alone"]=df["Family"]==1
df["Alone"]=df["Alone"].map({True: 1, False: 0})
df["Embarked"]=df["Embarked"].map({"C":0,"Q":1,"S":2})
df["Sex"]=df["Sex"].map({"male":0,"female":1})
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
x=df[["Sex","Age","Fare","Embarked","Alone","Family","Pclass"]]
y=df["Survived"]
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2,random_state=27)
#LogisticRegression
model1 = LogisticRegression(max_iter=1000)
model1.fit(x_train, y_train)
y_pred = model1.predict(x_test)
accuracy1=accuracy_score(y_test, y_pred)
print("Accuracy of the Logistic Regression model is: ", f"{accuracy1*100:.2f} %")
cm=confusion_matrix(y_test, y_pred)
print("Confusion Matrix:",cm)
print(classification_report(y_test, y_pred))
#DecisionTreeClassifier
model2 = DecisionTreeClassifier(random_state=27)
model2.fit(x_train, y_train)
y_pred = model2.predict(x_test)
accuracy2=accuracy_score(y_test, y_pred)
print("Accuracy of the Decision Tree model is: ", f"{accuracy2*100:.2f} %")
#RandomForestClassifier
model3 = RandomForestClassifier(random_state=27)
model3.fit(x_train, y_train)
y_pred = model3.predict(x_test)
accuracy3=accuracy_score(y_test, y_pred)
print("Accuracy of the Random Forest model is: ", f"{accuracy3*100:.2f} %")
#KNeighborsClassifier
model4 = KNeighborsClassifier()
model4.fit(x_train, y_train)
y_pred = model4.predict(x_test)
accuracy4=accuracy_score(y_test, y_pred)
print("Accuracy of the K-Nearest Neighbors model is: ", f"{accuracy4*100:.2f} %")
#comparing the models
models = ["Logistic", "Decision Tree", "Random Forest", "KNN"]
scores = [
    accuracy1 * 100,
    accuracy2 * 100,
    accuracy3 * 100,
    accuracy4 * 100
]
plt.bar(models, scores, color=["blue", "green", "red", "brown"])
plt.ylabel("Accuracy (%)")
plt.title("Model Comparison")
plt.show(block=False)
print("Feature Importances (Random Forest):")
for i in range(len(x.columns)):
    print(x.columns[i], ":", f"{model3.feature_importances_[i]*100:4.2f}%")
new_prediction =input("enter if you want to predict more 'yes' or 'no':").lower()
while new_prediction =="yes":
             new_passenger = pd.DataFrame({
                        "Sex": [int(input("Enter the sex (0 for male, 1 for female): "))],
                        "Age": [int(input("Enter the age: "))],
                        "Fare": [float(input("Enter the fare: "))],
                        "Embarked": [int(input("Enter the embarked (0 for C, 1 for Q, 2 for S): "))],
                        "Alone": [int(input("Enter if alone (1 for yes, 0 for no): "))],
                        "Family": [int(input("Enter the family size: "))],
                        "Pclass": [int(input("Enter the passenger class (1, 2, or 3): "))]})
             prediction = model3.predict(new_passenger)
             if prediction[0]== 1:
               print("The passenger is predicted to survive.")
             else:
               print("The passenger is predicted to not survive.")
             new_prediction = input("Do you want to predict another passenger? (yes/no): ").lower()
if new_prediction == "no":
  print("Thank you for using the Titanic Survival Prediction model.")