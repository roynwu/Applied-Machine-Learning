# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

"""# Adaboost-SAMME"""

import numpy as np
import math
from sklearn import tree

class BoostedDT:

    def __init__(self, numBoostingIters=100, maxTreeDepth=3):
        '''
        Constructor

        Class Fields 
        clfs : List object containing individual DecisionTree classifiers, in order of creation during boosting
        betas : List of beta values, in order of creation during boosting
        '''

        self.clfs = None  # keep the class fields, and be sure to keep them updated during boosting
        self.betas = None 
        
        #TODO
        self.numBoostingIters = numBoostingIters
        self.maxTreeDepth = maxTreeDepth
        self.K = None # num of classes
        self.classes = None # list of unique classes


    def fit(self, X, y, random_state=None):
        '''
        Trains the model. 
        Be sure to initialize all individual Decision trees with the provided random_state value if provided.
        
        Arguments:
            X is an n-by-d Pandas Data Frame
            y is an n-by-1 Pandas Data Frame
            random_seed is an optional integer value
        '''
        #TODO

        # convert X,y to np array
        X = np.array(X)
        y = np.array(y).flatten()

        # initiate class instances
        self.K = np.unique(y, axis=0).size
        self.classes = np.unique(y, axis=0)
        self.clfs = []
        self.betas = []

        # 1. initialize vector of n uniform weights
        n = X.shape[0] # num of instances
        sample_weights = np.array([1.0/n for i in range(n)])

        # 2. loop through T boosting iterations
        for t in range(self.numBoostingIters):

          # 3. train model on X,y with weights
          h_t = tree.DecisionTreeClassifier(max_depth=self.maxTreeDepth, 
                                            random_state=random_state)
          h_t = h_t.fit(X, y, sample_weight=sample_weights)
          self.clfs.append(h_t)

          # 4. compute weighted training error rate
          y_pred = h_t.predict(X)
          idx = np.where(y_pred != y)
          error_t = np.sum(sample_weights[idx])

          # 5. calculate beta
          beta_t = 0.5 * (np.log((1 - error_t) / error_t) + np.log(self.K - 1))
          self.betas.append(beta_t)

          # 6. update instance weights
          for i in range(n):
            if y_pred[i] == y[i]:
              sample_weights[i] = sample_weights[i] * np.exp(-1 * beta_t)
            elif y_pred[i] != y[i]:
              sample_weights[i] = sample_weights[i] * np.exp(beta_t)

          # 7. normalize
          sample_weights = sample_weights / np.sum(sample_weights)


    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is an n-by-d Pandas Data Frame
        Returns:
            an n-by-1 Pandas Data Frame of the predictions
        '''
        #TODO

        # initialize beta matrix
        X = np.array(X)
        n = X.shape[0]
        beta_mat = np.zeros((n, self.K))
        
        # compute hypothesis
        for i in range(self.numBoostingIters):
            y_pred = self.clfs[i].predict(X)
            for j in range(self.K):
                beta_mat[:,j] += (y_pred == self.classes[j]) * self.betas[i]
          
        return pd.DataFrame(self.classes[np.argmax(beta_mat, axis=1)])

"""# Test BoostedDT"""

# import numpy as np
# from sklearn import datasets
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split

# def test_boostedDT():

#   # load the data set
#   sklearn_dataset = datasets.load_breast_cancer()
#   # convert to pandas df
#   df = pd.DataFrame(sklearn_dataset.data,columns=sklearn_dataset.feature_names)
#   df['CLASS'] = pd.Series(sklearn_dataset.target)
#   df.head()

#   # split randomly into training/testing
#   train, test = train_test_split(df, test_size=0.5, random_state=42)
#   # Split into X,y matrices
#   X_train = train.drop(['CLASS'], axis=1)
#   y_train = train['CLASS']
#   X_test = test.drop(['CLASS'], axis=1)
#   y_test = test['CLASS']


#   # train the decision tree
#   modelDT = DecisionTreeClassifier()
#   modelDT.fit(X_train, y_train)

#   # train the boosted DT
#   modelBoostedDT = BoostedDT(numBoostingIters=100, maxTreeDepth=2)
#   modelBoostedDT.fit(X_train, y_train)

#   # train sklearn's implementation of Adaboost
#   modelSKBoostedDT = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=100)
#   modelSKBoostedDT.fit(X_train, y_train)

#   # output predictions on the test data
#   ypred_DT = modelDT.predict(X_test)
#   ypred_BoostedDT = modelBoostedDT.predict(X_test)
#   ypred_SKBoostedDT = modelSKBoostedDT.predict(X_test)

#   # compute the training accuracy of the model
#   accuracy_DT = accuracy_score(y_test, ypred_DT)
#   accuracy_BoostedDT = accuracy_score(y_test, ypred_BoostedDT)
#   accuracy_SKBoostedDT = accuracy_score(y_test, ypred_SKBoostedDT)

#   print("Decision Tree Accuracy = "+str(accuracy_DT))
#   print("My Boosted Decision Tree Accuracy = "+str(accuracy_BoostedDT))
#   print("Sklearn's Boosted Decision Tree Accuracy = "+str(accuracy_SKBoostedDT))
#   print()
#   print("Note that due to randomization, your boostedDT might not always have the ")
#   print("exact same accuracy as Sklearn's boostedDT.  But, on repeated runs, they ")
#   print("should be roughly equivalent and should usually exceed the standard DT.")

# test_boostedDT()

"""# Challenge: Generalizing to Unseen Data"""

# from google.colab import drive
# drive.mount('/content/drive')

# import pandas as pd
# import numpy as np

# # Load all data tables
# baseDir = '/content/drive/My Drive/'
# data_df = pd.read_csv(baseDir + 'ChocolatePipes_trainData.csv')
# labels_df = pd.read_csv(baseDir + 'ChocolatePipes_trainLabels.csv')
# # df = pd.concat([data_df, labels_df], axis=1, sort=False)
# df = pd.merge(data_df, labels_df, how='inner', on='id',
#          left_index=False, right_index=False, sort=False)

# # Output debugging info
# print(data_df.shape)
# print(labels_df.shape)
# print(df.shape)
# # data_df.head()
# # labels_df.head()
# df.head()

# # Print information about the dataset
# print('Percentage of instances with missing features:')
# print(df.isnull().sum(axis=0)/df.shape[0])
# print()
# print('Class information:')
# print(df['label'].value_counts())

# # Print number of unique values for each column
# i = 0
# for col in df.columns:
#   print(str(i) + '. ' + col + ' unique values: ')
#   print(df[col].nunique())
#   print('-----')
#   i += 1

# # Get value counts for each column
# pd.set_option('display.max_rows', None)
# i = 0
# for col in df.columns:
#   if col in ['Country of factory', 'id', 'Chocolate consumers in town', 
#              'Lattitude', 'longitude', 'Height of pipe']:
#     # i += 1
#     continue
#   print(str(i) + '. ' + col + ' information: ')
#   print(df[col].value_counts())
#   print('-----')
#   i += 1

"""### **Cleaning Data Process:**

1. **Create list to replace missing values**
2. **Drop** \
'id' \
'Recorded by'

3. **Convert to binary** \
'Official or Unofficial pipe' \
'Does factory offer tours'

4. **Convert to cat codes** \
'management_group' \
'Region code' \
'District code' \
'Oompa loompa management' \
  **Additional (needs more cleaning)** \
'Country of factory' \
'Country funded by' \
'oompa loomper' \

5. **Manipulate dates** \
'Date of entry'
"""

# # Make copy of df
# df1 = df.copy()

# # Create list of values to replace missing values
# fill_lst = []
# for col in df1.columns:
#   fill = df1[col].mode()[0]
#   fill_lst.append(fill) #

# # Get cat codes
# mgroup_catcodes = df1['management_group'].astype('category').cat.codes #
# region_catcodes = df1['Region code'].astype('category').cat.codes #
# district_catcodes = df1['District code'].astype('category').cat.codes #
# olmanagement_catcodes = df1['Oompa loompa management'].astype('category').cat.codes #

# # Clean 'oompa loomper'
# ol_count = df1['oompa loomper'].value_counts()
# ol_replace = ol_count[ol_count < 5]
# # print(ol_replace.sum(axis=0, skipna=True))
# # print(len(ol_replace))
# # print(len(ol_count))
# # print(len(ol_count) - len(ol_replace))
# ol_replace_lst = ol_replace.index.tolist()
# for ol in ol_replace_lst: 
#   df1.loc[(df1['oompa loomper'] == ol), 'oompa loomper'] = -1 
# ol_catcodes = df1['oompa loomper'].astype('category').cat.codes #
# # Test
# # df1['oompa loomper'] = ol_catcodes
# # df1['oompa loomper'].value_counts()

# # Clean 'Country of factory'
# factory_count = df1['Country of factory'].value_counts()
# factory_replace = factory_count[factory_count == 1]
# # print(factory_replace.sum(axis=0, skipna=True))
# # print(len(factory_replace))
# # print(len(factory_count))
# # print(len(factory_count) - len(factory_replace))
# factory_replace_lst = factory_replace.index.tolist()
# for factory in factory_replace_lst:
#   df1.loc[(df1['Country of factory'] == factory), 'Country of factory'] = -1
# factory_catcodes = df1['Country of factory'].astype('category').cat.codes # 
# # Test
# # df1['Country of factory'] = factory_catcodes
# # df1['Country of factory'].value_counts()

# # Clean 'Country funded by'
# funder_count = df1['Country funded by'].value_counts()
# funder_replace = funder_count[funder_count < 5]
# # print(funder_replace.sum(axis=0, skipna=True))
# # print(len(funder_replace))
# # print(len(funder_count))
# # print(len(funder_count) - len(funder_replace))
# funder_replace_lst = funder_replace.index.tolist() 
# for funder in funder_replace_lst: 
#   df1.loc[(df1['Country funded by'] == funder), 'Country funded by'] = -1 
# funder_catcodes = df1['Country funded by'].astype('category').cat.codes # 
# # Test
# # df1['Country funded by'] = funder_catcodes
# # df1['Country funded by'].value_counts()

# # Store cat codes in list
# catcode_lst = [mgroup_catcodes, region_catcodes, 
#                district_catcodes, olmanagement_catcodes,
#                ol_catcodes, factory_catcodes, funder_catcodes]

# # Store numerical columns in list for standardizing
# numeric_lst = ['Size of chocolate pool', 'Height of pipe', 'Chocolate consumers in town', 'longitude', 'Lattitude']

# def preprocess_df(input_df, fill_lst, catcode_lst, numeric_lst):
#     from sklearn import preprocessing

#     # 1. Fill missing values
#     for i in range(len(input_df.columns)):
#       input_df[input_df.columns[i]].fillna(fill_lst[i], inplace=True)

#     # 2. Drop
#     input_df.drop(['id','Recorded by'], axis=1, inplace=True)

#     # 3. Convert to binary
#     lb = preprocessing.LabelBinarizer()
#     for binary_feature in ['Official or Unofficial pipe', 'Does factory offer tours']:
#       input_df[binary_feature] = lb.fit_transform(input_df[binary_feature])

#     # 4. Convert to cat codes
#     input_df['management_group'] = catcode_lst[0]
#     input_df['Region code'] = catcode_lst[1]
#     input_df['District code'] = catcode_lst[2]
#     input_df['Oompa loompa management'] = catcode_lst[3]
#     input_df['oompa loomper'] = catcode_lst[4]
#     input_df['Country of factory'] = catcode_lst[5]
#     input_df['Country funded by'] = catcode_lst[6]

#     # 5. Manipulate dates
#     input_df['Date of entry'] = pd.to_datetime(input_df['Date of entry'])
#     input_df['Month'] = input_df.apply(lambda x: x['Date of entry'].month, axis=1)
#     input_df['Day'] = input_df.apply(lambda x: x['Date of entry'].day, axis=1)
#     input_df['Year'] = input_df.apply(lambda x: x['Date of entry'].year, axis=1)
#     input_df.drop('Date of entry', axis=1, inplace=True)

#     # Standardize 
#     scaler = preprocessing.StandardScaler()
#     # input_df[numeric_lst] = scaler.fit_transform(input_df[numeric_lst])
#     input_df = scaler.fit_transform(input_df)

#     output_df = np.asarray(input_df)
  
#     return output_df

# # Sanity check
# input_test1 = data_df.copy()
# input_test2 = preprocess_df(input_test1, fill_lst, catcode_lst, numeric_lst)
# input_test2.shape

"""### Train/Test Models"""

# # Separate df into features and labels
# X = df.drop('label', axis=1)
# y = df['label'].values

# # Preprocess X
# X = preprocess_df(X, fill_lst, catcode_lst, numeric_lst)

# # Create training/test sets
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# import random
# import numpy as np
# from sklearn.metrics import accuracy_score
# from sklearn import tree
# from sklearn.tree import DecisionTreeClassifier

# def cross_validated_accuracy(Classifier, X, y, num_trials, num_folds, random_seed):
#   random.seed(random_seed)
#   """
#    Args:
#         DecisionTreeClassifier: An Sklearn DecisionTreeClassifier (e.g., created by "tree.DecisionTreeClassifier(criterion='entropy')")
#         X: Input features
#         y: Labels
#         num_trials: Number of trials to run of cross validation
#         num_folds: Number of folds (the "k" in "k-folds")
#         random_seed: Seed for uniform execution (Do not change this) 

#     Returns:
#         cvScore: The mean accuracy of the cross-validation experiment
#   """

#   # Combine X,y
#   X = np.array(X)
#   y = np.array(y)
#   dataset = np.concatenate((X,y.reshape(-1,1)),axis=1)

#   # Store each cv score in list
#   scores = []

#   # Loop through trials
#   for i in range(num_trials):

#     # Shuffle dataset
#     np.random.shuffle(dataset)

#     # Split dataset into folds
#     data_split = np.array_split(dataset, num_folds)

#     # Loop through folds
#     for k in range(len(data_split)):
#       # Test set
#       test_set = data_split[k]

#       # Train set
#       train_lst = []
#       for i in range(len(data_split)):
#         if i != k:
#           train_lst.append(data_split[i])
#       train_set = np.concatenate(train_lst,axis=0)

#       # Split into X,y
#       X_train = train_set[:,:-1]
#       y_train = train_set[:,-1]
#       X_test = test_set[:,:-1]
#       y_test = test_set[:,-1]

#       # Use clf from parameter and record score
#       clf = Classifier
#       clf.fit(X_train,y_train)
#       y_pred = clf.predict(X_test)
#       scores.append(accuracy_score(y_test,y_pred))

#   # Average
#   cvScore = np.mean(scores)

#   return cvScore

# import numpy as np
# from sklearn import tree
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score

# # train the decision tree
# modelDT = DecisionTreeClassifier()
# modelDT.fit(X_train, y_train)

# # train the boosted DT
# modelBoostedDT = BoostedDT(numBoostingIters=100, maxTreeDepth=2)
# modelBoostedDT.fit(X_train, y_train)

# # train sklearn's implementation of Adaboost
# modelSKBoostedDT = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=100)
# modelSKBoostedDT.fit(X_train, y_train)

# # output predictions on the test data
# ypred_DT = modelDT.predict(X_test)
# ypred_BoostedDT = modelBoostedDT.predict(X_test)
# ypred_SKBoostedDT = modelSKBoostedDT.predict(X_test)

# # # Use cross-validation on the training data to get an estimate of the performance on unseen (test) data
# # from sklearn.model_selection import cross_val_score, StratifiedKFold
# # from sklearn import metrics
# # # Set number of trials and number of folds
# # NUM_TRIALS = 10
# # NUM_FOLDS = 10
# # # Array to store scores
# # cVscore_DT = np.zeros(NUM_TRIALS * NUM_FOLDS)
# # cVscore_SKBoostedDT = np.zeros(NUM_TRIALS * NUM_FOLDS)  
# # # Loop for each trial
# # for i in range(NUM_TRIALS):
# #   cVscore_DT[i*NUM_FOLDS : (i+1)*NUM_FOLDS] = cross_val_score(modelDT, X_train, y_train, 
# #                                                               cv=StratifiedKFold(NUM_FOLDS), scoring='accuracy')
# #   cVscore_SKBoostedDT[i*NUM_FOLDS : (i+1)*NUM_FOLDS] = cross_val_score(modelSKBoostedDT, X_train, y_train, 
# #                                                                        cv=StratifiedKFold(NUM_FOLDS), scoring='accuracy')
# # # cvScore_DT = cross_validated_accuracy(Classifier=modelDT, X=X_train, y=y_train, 
# # #                                       num_trials=10, num_folds=10, random_seed=42)
# # cvScore_BoostedDT = cross_validated_accuracy(Classifier=modelBoostedDT, X=X_train, y=y_train, 
# #                                              num_trials=10, num_folds=10, random_seed=42)
# # # cvScore_SKBoostedDT = cross_validated_accuracy(Classifier=modelSKBoostedDT, X=X_train, y=y_train, 
# # #                                                num_trials=10, num_folds=10, random_seed=42)

# # compute the training accuracy of the model
# accuracy_DT = accuracy_score(y_test, ypred_DT)
# accuracy_BoostedDT = accuracy_score(y_test, ypred_BoostedDT)
# accuracy_SKBoostedDT = accuracy_score(y_test, ypred_SKBoostedDT)

# print("Decision Tree Accuracy = "+str(accuracy_DT))
# print("My Boosted Decision Tree Accuracy = "+str(accuracy_BoostedDT))
# print("Sklearn's Boosted Decision Tree Accuracy = "+str(accuracy_SKBoostedDT))
# # print('')
# # print("Decision Tree Generalization Accuracy = "+str(cvScore_DT.mean()))
# # print("My Boosted Decision Tree Generalization Accuracy = "+str(cvScore_BoostedDT))
# # print("Sklearn's Boosted Decision Tree Generalization Accuracy = "+str(cvScore_SKBoostedDT.mean()))

# from sklearn import svm

# # train the SVM
# modelSVM = svm.SVC(C=10.0, decision_function_shape='ovo', probability=True)
# modelSVM.fit(X_train, y_train)

# # output predictions on the test data
# ypred_SVM = modelSVM.predict(X_test)

# # # Use cross-validation on the training data to get an estimate of the performance on unseen (test) data
# # from sklearn.model_selection import cross_val_score, StratifiedKFold
# # from sklearn import metrics
# # # Set number of trials and number of folds
# # NUM_TRIALS = 10
# # NUM_FOLDS = 10
# # # Array to store scores
# # cVscore_SVM = np.zeros(NUM_TRIALS * NUM_FOLDS)
# # # Loop for each trial
# # for i in range(NUM_TRIALS):
# #   cVscore_SVM[i*NUM_FOLDS : (i+1)*NUM_FOLDS] = cross_val_score(modelSVM, X_train, y_train, 
# #                                                               cv=StratifiedKFold(NUM_FOLDS), scoring='accuracy')

# # compute the training accuracy of the model
# accuracy_SVM = accuracy_score(y_test, ypred_SVM)

# print("SVM Accuracy = "+str(accuracy_SVM))
# # print("SVM Generalization Accuracy = "+str(cvScore_SVM.mean()))

# from sklearn.linear_model import LogisticRegression

# # train the Random Forest
# modelLRC = LogisticRegression()
# modelLRC.fit(X_train, y_train)

# # output predictions on the test data
# ypred_LRC = modelLRC.predict(X_test)

# # # Use cross-validation on the training data to get an estimate of the performance on unseen (test) data
# # from sklearn.model_selection import cross_val_score, StratifiedKFold
# # from sklearn import metrics
# # # Set number of trials and number of folds
# # NUM_TRIALS = 10
# # NUM_FOLDS = 10
# # # Array to store scores
# # cVscore_LRC = np.zeros(NUM_TRIALS * NUM_FOLDS)
# # # Loop for each trial
# # for i in range(NUM_TRIALS):
# #   cVscore_LRC[i*NUM_FOLDS : (i+1)*NUM_FOLDS] = cross_val_score(modelLRC, X_train, y_train, 
# #                                                               cv=StratifiedKFold(NUM_FOLDS), scoring='accuracy')

# # compute the training accuracy of the model
# accuracy_LRC = accuracy_score(y_test, ypred_LRC)

# print("Logistic Regression Accuracy = "+str(accuracy_LRC))
# # print("Logistic Regression Generalization Accuracy = "+str(cvScore_LRC.mean()))

# from sklearn.ensemble import RandomForestClassifier

# # train the Random Forest
# modelRFC = RandomForestClassifier(n_estimators=600, class_weight='balanced')
# modelRFC.fit(X_train, y_train)

# # output predictions on the test data
# ypred_RFC = modelRFC.predict(X_test)

# # # Use cross-validation on the training data to get an estimate of the performance on unseen (test) data
# # from sklearn.model_selection import cross_val_score, StratifiedKFold
# # from sklearn import metrics
# # # Set number of trials and number of folds
# # NUM_TRIALS = 10
# # NUM_FOLDS = 10
# # # Array to store scores
# # cVscore_RFC = np.zeros(NUM_TRIALS * NUM_FOLDS)
# # # Loop for each trial
# # for i in range(NUM_TRIALS):
# #   cVscore_RFC[i*NUM_FOLDS : (i+1)*NUM_FOLDS] = cross_val_score(modelRFC, X_train, y_train, 
# #                                                               cv=StratifiedKFold(NUM_FOLDS), scoring='accuracy')

# # compute the training accuracy of the model
# accuracy_RFC = accuracy_score(y_test, ypred_RFC)

# print("Random Forest Accuracy = "+str(accuracy_RFC))
# # print("Random Forest Generalization Accuracy = "+str(cvScore_RFC.mean()))

# from sklearn.neural_network import MLPClassifier

# # train the MLP
# modelMLP = MLPClassifier(hidden_layer_sizes=(100,50), activation='relu', solver='adam', alpha=0.01, 
#                     batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, 
#                     max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, 
#                     momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, 
#                     beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)
# modelMLP.fit(X_train, y_train)

# # output predictions on the test data
# ypred_MLP = modelMLP.predict(X_test)

# # # Use cross-validation on the training data to get an estimate of the performance on unseen (test) data
# # from sklearn.model_selection import cross_val_score, StratifiedKFold
# # from sklearn import metrics
# # # Set number of trials and number of folds
# # NUM_TRIALS = 10
# # NUM_FOLDS = 10
# # # Array to store scores
# # cVscore_MLP = np.zeros(NUM_TRIALS * NUM_FOLDS)
# # # Loop for each trial
# # for i in range(NUM_TRIALS):
# #   cVscore_MLP[i*NUM_FOLDS : (i+1)*NUM_FOLDS] = cross_val_score(modelMLP, X_train, y_train, 
# #                                                               cv=StratifiedKFold(NUM_FOLDS), scoring='accuracy')

# # compute the training accuracy of the model
# accuracy_MLP = accuracy_score(y_test, ypred_MLP)

# print("MLP Accuracy = "+str(accuracy_MLP))
# # print("MLP Generalization Accuracy = "+str(cvScore_MLP.mean()))

# from sklearn.ensemble import VotingClassifier

# # train the Voting Classifier
# modelVOTING = VotingClassifier(estimators=[('SVM', modelSVM), ('RFC', modelRFC), ('MLP', modelMLP)], 
#                                voting='soft', weights=[1,2,1])
# modelVOTING.fit(X_train, y_train)

# # output predictions on the test data
# ypred_VOTING = modelVOTING.predict(X_test)

# # # Use cross-validation on the training data to get an estimate of the performance on unseen (test) data
# # from sklearn.model_selection import cross_val_score, StratifiedKFold
# # from sklearn import metrics
# # # Set number of trials and number of folds
# # NUM_TRIALS = 10
# # NUM_FOLDS = 10
# # # Array to store scores
# # cVscore_VOTING = np.zeros(NUM_TRIALS * NUM_FOLDS)
# # # Loop for each trial
# # for i in range(NUM_TRIALS):
# #   cVscore_VOTING[i*NUM_FOLDS : (i+1)*NUM_FOLDS] = cross_val_score(modelVOTING, X_train, y_train, 
# #                                                               cv=StratifiedKFold(NUM_FOLDS), scoring='accuracy')

# # compute the training accuracy of the model
# accuracy_VOTING = accuracy_score(y_test, ypred_VOTING)

# print("Voting Classifier Accuracy = "+str(accuracy_VOTING))
# # print("Voting Classifier Generalization Accuracy = "+str(cvScore_VOTING.mean()))

"""### Test on Unlabeled Data"""

# # Read data and save as pd
# leaderboard_df = pd.read_csv(baseDir + 'ChocolatePipes_leaderboardTestData.csv')
# grading_df = pd.read_csv(baseDir + 'ChocolatePipes_gradingTestData.csv')
# print(leaderboard_df.shape)
# print(grading_df.shape)

# # Preprocess unlabeled data
# leaderboard_data = preprocess_df(leaderboard_df, fill_lst, catcode_lst, numeric_lst)
# grading_data = preprocess_df(grading_df, fill_lst, catcode_lst, numeric_lst)
# print(leaderboard_data.shape)
# print(grading_data.shape)

# # Predict labels
# ypred_BoostedDT_leaderboard = modelBoostedDT.predict(leaderboard_data)
# ypred_BoostedDT_grading = modelBoostedDT.predict(grading_data)
# ypred_SVC_leaderboard = modelSVM.predict(leaderboard_data)
# ypred_SVC_grading = modelSVM.predict(grading_data)
# ypred_best_leaderboard = modelVOTING.predict(leaderboard_data)
# ypred_best_grading = modelVOTING.predict(grading_data)
# # ypred_best_leaderboard = modelRFC.predict(leaderboard_data)
# # ypred_best_grading = modelRFC.predict(grading_data)

# # Save to csv file
# np.savetxt('predictions-leaderboard-BoostedDT.csv', ypred_BoostedDT_leaderboard, delimiter='\n')
# np.savetxt('predictions-grading-BoostedDT.csv', ypred_BoostedDT_grading, delimiter='\n')
# np.savetxt('predictions-leaderboard-SVC.csv', ypred_SVC_leaderboard, delimiter='\n')
# np.savetxt('predictions-grading-SVC.csv', ypred_SVC_grading, delimiter='\n')
# np.savetxt('predictions-leaderboard-best.csv', ypred_best_leaderboard, delimiter='\n')
# np.savetxt('predictions-grading-best.csv', ypred_best_grading, delimiter='\n')

# ypred_best_leaderboard_df = pd.read_csv('predictions-leaderboard-best.csv')
# ypred_best_grading_df = pd.read_csv('predictions-grading-best.csv')

# ypred_best_leaderboard_df['2.000000000000000000e+00'].value_counts()

# ypred_best_grading_df['0.000000000000000000e+00'].value_counts()

