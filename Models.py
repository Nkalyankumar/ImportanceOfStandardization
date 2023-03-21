from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression, Perceptron
from sklearn.metrics import r2_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
#Iris data  is used for Building models

#Accuracy of models
unpreprocessed_accuracies = []
preprocessed_accuracies = []
model_names = ["RegressionModel","Logistic Regression","Naive Bayes Classifier","Decision Tree Model","KNNClassifier Model","SupportVectorClassifier","Perceptron Model"]

#Loading the data from the datasets
iris = datasets.load_iris()
data = iris.data  
target = iris.target

#Splitting the data into training and testing data
data_train,data_test,target_train,target_test = train_test_split(data,target)

#Building RegressionModel
LiReModel =LinearRegression()
LiReModel.fit(data_train,target_train)
LiReModel_pred_target = LiReModel.predict(data_test)
LiReModel_accuracy = r2_score(target_test,LiReModel_pred_target)
unpreprocessed_accuracies.append(LiReModel_accuracy*100)

#Building Logistic Regression for classification
LRModel = LogisticRegression(random_state = True,max_iter = 1000)
LRModel.fit(data_train,target_train)
LRModel_pred_target = LRModel.predict(data_test)
LRModel_accuracy = accuracy_score(target_test,LRModel_pred_target)
unpreprocessed_accuracies.append(LRModel_accuracy*100)

#Building Naive Bayes Classifier Model
NBModel = GaussianNB()
NBModel.fit(data_train,target_train)
NBModel_pred_target = NBModel.predict(data_test)
NBModel_accuracy = accuracy_score(target_test,NBModel_pred_target)
unpreprocessed_accuracies.append(NBModel_accuracy*100)

#Building Decision Tree Model 
DTModel = DecisionTreeClassifier()
DTModel.fit(data_train,target_train)
DTModel_pred_target = DTModel.predict(data_test)
DTModel_accuracy = accuracy_score(target_test,DTModel_pred_target)
unpreprocessed_accuracies.append(DTModel_accuracy*100)

#Building KNNClassifier Model
KNNModel = KNeighborsClassifier(n_neighbors = 3)
KNNModel.fit(data_train,target_train)
KNNModel_pred_target = KNNModel.predict(data_test)
KNNModel_accuracy = accuracy_score(target_test,KNNModel_pred_target)
unpreprocessed_accuracies.append(KNNModel_accuracy*100)

#Building SVM model
SVCModel =  LinearSVC()
SVCModel.fit(data_train,target_train)
SVCModel_pred_target = SVCModel.predict(data_test)
SVCModel_accuracy = accuracy_score(target_test,SVCModel_pred_target)
unpreprocessed_accuracies.append(SVCModel_accuracy*100)

#Building Perceptron Model
PtModel =  Perceptron(random_state = True,max_iter = 100,alpha = 0.0001)
PtModel.fit(data_train,target_train)
PtModel_pred_target = PtModel.predict(data_test)
PtModel_accuracy = accuracy_score(target_test,PtModel_pred_target)
unpreprocessed_accuracies.append(PtModel_accuracy*100)


#Building the models
#Standardization of Iris data
stdsclr = StandardScaler()
stdsclr.fit(data_train)
data_train = stdsclr.transform(data_train)
data_test = stdsclr.transform(data_test)

#Building RegressionModel
LiReModel =LinearRegression()
LiReModel.fit(data_train,target_train)
LiReModel_pred_target = LiReModel.predict(data_test)
LiReModel_accuracy = r2_score(target_test,LiReModel_pred_target)
preprocessed_accuracies.append(LiReModel_accuracy*100)

#Building Logistic Regression for classification
LRModel = LogisticRegression(random_state = True,max_iter = 1000)
LRModel.fit(data_train,target_train)
LRModel_pred_target = LRModel.predict(data_test)
LRModel_accuracy = accuracy_score(target_test,LRModel_pred_target)
preprocessed_accuracies.append(LRModel_accuracy*100)

#Building Naive Bayes Classifier Model
NBModel = GaussianNB()
NBModel.fit(data_train,target_train)
NBModel_pred_target = NBModel.predict(data_test)
NBModel_accuracy = accuracy_score(target_test,NBModel_pred_target)
preprocessed_accuracies.append(NBModel_accuracy*100)

#Building Decision Tree Model 
DTModel = DecisionTreeClassifier()
DTModel.fit(data_train,target_train)
DTModel_pred_target = DTModel.predict(data_test)
DTModel_accuracy = accuracy_score(target_test,DTModel_pred_target)
preprocessed_accuracies.append(DTModel_accuracy*100)

#Building KNNClassifier Model
KNNModel = KNeighborsClassifier(n_neighbors = 3)
KNNModel.fit(data_train,target_train)
KNNModel_pred_target = KNNModel.predict(data_test)
KNNModel_accuracy = accuracy_score(target_test,KNNModel_pred_target)
preprocessed_accuracies.append(KNNModel_accuracy*100)

#Building SVM model
SVCModel =  LinearSVC()
SVCModel.fit(data_train,target_train)
SVCModel_pred_target = SVCModel.predict(data_test)
SVCModel_accuracy = accuracy_score(target_test,SVCModel_pred_target)
preprocessed_accuracies.append(SVCModel_accuracy*100)

#Building Perceptron Model
PtModel =  Perceptron(random_state = True,max_iter = 100,alpha = 0.0001)
PtModel.fit(data_train,target_train)
PtModel_pred_target = PtModel.predict(data_test)
PtModel_accuracy = accuracy_score(target_test,PtModel_pred_target)
preprocessed_accuracies.append(PtModel_accuracy*100)

#visualizing the results 
f = plt.figure()
f.set_figwidth(20)
f.set_figheight(12)
X_axis = np.arange(len(model_names))
plt.barh(X_axis - 0.2,unpreprocessed_accuracies, 0.4, label = 'Unpreprocessed')
plt.barh(X_axis + 0.2,preprocessed_accuracies, 0.4, label = 'preprocessed')
plt.yticks(X_axis, model_names)
plt.title("Accuracy_scores Without Standard Scaler")
plt.xlabel("Accuracy")
plt.ylabel("Model Used")
plt.legend()
plt.show()
plt.show()
