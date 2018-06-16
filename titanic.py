import numpy as np
import pandas as pd
from sklearn import datasets, svm, tree, preprocessing
import sklearn.ensemble as ske
import sys

PARAM_NUM = 8
USAGE = """%s <Class> <Sex> <Age> <Number of Siblings/Spouses Aboard> <Number of Parents/Children Aboard> <Fare> <Port of Embarkation>
	Class is the ticket's class (1, 2 or 3)
	Sex is the passenger's sex (M or F)
	Fare is the ticket's cost
	Port of Embarkation is 0 for Cherbourg, 1 for Queenstown and 2 for Southampton
"""

# preprocess the data: 
#	remove name, ticket number and cabin properties
#	drop all partially empty records
#	transform categorial features to numeric ones
def preprocess_titanic_df(df):
	processed_df = df.copy()
	le = preprocessing.LabelEncoder()
	processed_df = processed_df.dropna()
	processed_df.Sex = le.fit_transform(processed_df.Sex)
	processed_df.Embarked = le.fit_transform(processed_df.Embarked)
	processed_df = processed_df.drop(['Name','Ticket','Cabin'],axis=1)
	return processed_df
	
# sanitize the input from the user: transform m/f to 1/0 and cast everything from string to float
def sanitize_input(input_args):
	args = input_args[1:]
	args[1] = 0 if "f" == args[1].lower() else 1
	return [float(x) for x in args]
	
# given a classifier - fit to training data and predict result on test data
def classify(clf, X_train, y_train, X_test):
	clf.fit(X_train, y_train)
	return clf.predict(X_test)

def main(input_file):
	# validate arguments
	if len(sys.argv) != PARAM_NUM:
		print(USAGE % sys.argv[0])
		return
		
	# read data csv and preprocess
	titanic_df = pd.read_csv(input_file, na_values=['NA'])
	processed_df = preprocess_titanic_df(titanic_df)
	
	X = processed_df.drop(['Survived'], axis=1).values
	y = processed_df['Survived'].values
	
	# create a new sample from given user input
	x_test = np.asarray([X[-1][0] + 1] + sanitize_input(sys.argv)).reshape(1, -1)
	
	# helper function
	survive = lambda result: " not" if result == 0 else ""
	
	# get prediction from 3 types of algorithms
	clf_tree = tree.DecisionTreeClassifier(max_depth=10)
	result_tree = classify(clf_tree, X, y, x_test)[0]
	print("According to Tree classifier:\t\t You would%s survive the titanic!" % survive(result_tree))
	
	clf_svm = svm.SVC()
	result_svm = classify(clf_svm, X, y, x_test)[0]
	print("According to SVM classifier:\t\t You would%s survive the titanic!"  % survive(result_svm))
	
	clf_rf = ske.RandomForestClassifier(n_estimators=50)
	result_rf = classify(clf_rf, X, y, x_test)[0]
	print("According to RandomForest classifier:\t You would%s survive the titanic!" % survive(result_rf))
	
	clf_gb = ske.GradientBoostingClassifier(n_estimators=50)
	result_gb = classify(clf_gb, X, y, x_test)[0]
	print("According to Gradient Boost classifier:\t You would%s survive the titanic!" % survive(result_gb))
	
	
if __name__ == "__main__":
	main('/data/titanic.csv')