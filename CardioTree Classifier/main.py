import pandas as pd
import GWCutilities as util

pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)

print("\n-----\n")

#Create a variable to read the dataset
df = pd.read_csv("heartDisease_2020_sampling.csv")

print(
    "We will be performing data analysis on this Indicators of Heart Disease Dataset. Here is a sample of it: \n"
)

#Print the dataset's first five rows
print(df.head())

input("\n Press Enter to continue.\n")



#Data Cleaning
#Label encode the dataset
df = util.labelEncoder(df, ["HeartDisease", "GenHealth", "Smoking", "AlcoholDrinking", "PhysicalActivity", "AgeCategory"])

print("\nHere is a preview of the dataset after label encoding. \n")
print(df.head)

input("\nPress Enter to continue.\n")

#One hot encode the dataset
df = util.labelEncoder(df, ["Sex", "Race"])

print(
    "\nHere is a preview of the dataset after one hot encoding. This will be the dataset used for data analysis: \n"
)
print(df.head)

input("\nPress Enter to continue.\n")



#Creates and trains Decision Tree Model
from sklearn.model_selection import train_test_split
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
print(X_train.head())

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth = 9, class_weight = "balanced")
clf = clf.fit(X_train, y_train)




#Test the model with the testing data set and prints accuracy score
test_predictions = clf.predict(X_test)

from sklearn.metrics import accuracy_score
test_acc = accuracy_score(y_test, test_predictions)

print("The accuracy with the testing data set of the Decision Tree is: " + str(test_acc))

#Prints the confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, test_predictions, labels = [1,0])
print("The confusion matrix of the tree is: ")
print(cm)

#Test the model with the training data set and prints accuracy score
train_predictions = clf.predict(X_train)

from sklearn.metrics import accuracy_score
train_acc = accuracy_score(y_train, train_predictions)

print("The accuracy with the testing data set of the Decision Tree is: " + str(train_acc))



input("\nPress Enter to continue.\n")



#Prints another application of Decision Trees and considerations
print("\n One application of decision trees in another field is in the field of ecology to identify invasive/harmful species vs their harmless lookalikes. One factor we would have to consider is the weight of the training data to make sure the model isn't skewed towards one class. Also, we have to consider the ways we encode the data to make sure we don't imply any bias. Lastly, we need to adjust the maximum depth of the decision tree to ensure we have the best results.")





#Prints a text representation of the Decision Tree
print("\nBelow is a text representation of how the Decision Tree makes choices:\n")
input("\nPress Enter to continue.\n")

util.printTree(clf, X.columns)
