import random
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

def cross_validated_accuracy(DecisionTreeClassifier, X, y, num_trials, num_folds, random_seed):
  random.seed(random_seed)
  """
   Args:
        DecisionTreeClassifier: An Sklearn DecisionTreeClassifier (e.g., created by "tree.DecisionTreeClassifier(criterion='entropy')")
        X: Input features
        y: Labels
        num_trials: Number of trials to run of cross validation
        num_folds: Number of folds (the "k" in "k-folds")
        random_seed: Seed for uniform execution (Do not change this) 

    Returns:
        cvScore: The mean accuracy of the cross-validation experiment

    Notes:
        1. You may NOT use the cross-validation functions provided by Sklearn
  """
  ## TODO ##

  # Combine X,y
  X = np.array(X)
  y = np.array(y)
  dataset = np.concatenate((X,y.reshape(-1,1)),axis=1)
  # Store each cv score in list
  scores = []

  # Loop through trials
  for i in range(num_trials):
    # Shuffle dataset
    # random.seed(random_seed)
    # random.shuffle(dataset)
    np.random.shuffle(dataset)
    
    # data_split = []
    # data_copy = list(dataset)
    # fold_size = int(len(dataset) / num_folds)
    # test_size = 1 / num_folds

    # for j in range(num_folds):
    #   fold = []

    #   while len(fold) < fold_size:
    #     index = random.randrange(len(data_copy))
    #     fold.append(data_copy.pop(index))

    #   data_split.append(fold)

    # data_split = np.array(data_split)

    # Split dataset into folds
    data_split = np.array_split(dataset, num_folds)

    # Loop through folds
    for k in range(len(data_split)):
      
      # Test set
      test_set = data_split[k]

      # Train set
      train_lst = []
      for i in range(len(data_split)):
        if i != k:
          train_lst.append(data_split[i])
      train_set = np.concatenate(train_lst,axis=0)

      # Split into X,y
      X_train = train_set[:,:-1]
      y_train = train_set[:,-1]
      X_test = test_set[:,:-1]
      y_test = test_set[:,-1]

      # Use clf from parameter and record score
      dtree = DecisionTreeClassifier
      dtree.fit(X_train,y_train)
      y_pred = dtree.predict(X_test)

      scores.append(accuracy_score(y_test,y_pred))

  # Average
  cvScore = np.mean(scores)

  # Confidence Interval (99%)
  x_bar = np.mean(scores)
  S = np.std(scores)
  n = len(scores)
  t = 2.626
  
  endpoint = t * (S / np.sqrt(n))

  # return cvScore, endpoint
  return cvScore


def automatic_dt_pruning(DecisionTreeClassifier, X, y, num_trials, num_folds, random_seed):
  random.seed(random_seed)
  """
  Returns the pruning parameter (i.e., ccp_alpha) with the highest cross-validated accuracy

  Args:
        DecisionTreeClassifier  : An Sklearn DecisionTreeClassifier (e.g., created by "tree.DecisionTreeClassifier(criterion='entropy')")      
        X (Pandas.DataFrame)    : Input Features
        y (Pandas.Series)       : Labels
        num_trials              : Number of trials to run of cross validation
        num_folds               : Number of folds for cross validation (The "k" in "k-folds") 
        random_seed             : Seed for uniform execution (Do not change this)


    Returns:
        ccp_alpha : Tuned pruning paramter with highest cross-validated accuracy

    Notes:
        1. Don't change any other Decision Tree Classifier parameters other than ccp_alpha
        2. Use the cross_validated_accuracy function you implemented to find the cross-validated accuracy

  """

  ## TODO ##

  X = np.array(X)
  y = np.array(y)

  dtree = DecisionTreeClassifier
  path = dtree.cost_complexity_pruning_path(X, y)
  ccp_alphas, impurities = path.ccp_alphas, path.impurities
  # print(ccp_alphas)

  scores = {}
  for ccp_alpha in ccp_alphas[1:-1]:
    clf = tree.DecisionTreeClassifier(random_state=random_seed, ccp_alpha=ccp_alpha,class_weight='balanced')
    # cvScore, endpoint = cross_validated_accuracy(clf, X, y, num_trials, num_folds, random_seed)
    cvScore = cross_validated_accuracy(clf, X, y, num_trials, num_folds, random_seed)
    scores[ccp_alpha] = cvScore

  # # Record scores in dict
  # scores = {}
  # # List of ccp_alphas to try
  # ccp_alphas = np.linspace(0.01, 0.05, 30)

  # # Loop through ccp_alpha
  # for ccp_alpha in ccp_alphas:
  #   # Initiate clf and record k-fold cv score
  #   clf = tree.DecisionTreeClassifier(random_state=random_seed, ccp_alpha=ccp_alpha)
  #   cvScore, endpoint = cross_validated_accuracy(clf, X, y, num_trials, num_folds, random_seed)
  #   scores[ccp_alpha] = cvScore
  
  # Return key with max value
  ccp_alpha = max(scores, key=scores.get)

  return ccp_alpha
