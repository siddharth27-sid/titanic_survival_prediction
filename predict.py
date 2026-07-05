import pickle
# load the trained model
with open("titanic_model.pkl", "rb") as file:
    model3 = pickle.load(file)
#predicting the survival of a new passenger
new_prediction =input("enter if you want to predict more 'yes' or 'no':").lower()
while new_prediction =="yes":
             new_passenger = [[int(input("Enter the sex (0 for male, 1 for female): ")),
                   int(input("Enter the age: ")),
                    int(input("Enter the fare: ")),
                     int(input("Enter the embarked (0 for C, 1 for Q, 2 for S): ")),
                       int(input("Enter if alone (1 for yes, 0 for no): ")),
                         int(input("Enter the family size: ")),
                           int(input("Enter the passenger class (1, 2, or 3): "))]]
             prediction = model3.predict(new_passenger)
             if prediction[0]== 1:
               print("The passenger is predicted to survive.")
             else:
               print("The passenger is predicted to not survive.")
             new_prediction = input("Do you want to predict another passenger? (yes/no): ").lower()
if new_prediction == "no":
  print("Thank you for using the Titanic Survival Prediction model.")