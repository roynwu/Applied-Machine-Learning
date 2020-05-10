# Applied-Machine-Learning
This repository contains a collection of notebooks (assignments from CIS 519) covering derivations and applications of the most widely-used algorithms in machine learning.  
Main libraries used are [PyTorch](https://github.com/pytorch/pytorch), [Scikit-Learn](https://github.com/scikit-learn/scikit-learn), [NumPy](https://github.com/numpy/numpy), [Pandas](https://github.com/pandas-dev/pandas), and [Open AI Gym](https://github.com/openai/gym).

## Getting Started
To install PyTorch, see installation instructions on the [PyTorch website](pytorch.org).  
To install Scikit-Learn, see installation instructions on the [Scikit-Learn website](scikit-learn.org).  
To install NumPy, see installation instructions on the [NumPy website](numpy.org).  
To install Pandas, see installation instructions on the [Pandas website](pandas.pydata.org).    
To install Gym, see installation instructions on the [Open AI Gym website](https://gym.openai.com/).

## Topics

* 1 - [Decision Trees](https://github.com/roynwu/Applied-Machine-Learning/blob/master/1.%20Decision%20Trees/notebook/decision_tree.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ifxNBdChd_8kqgWBxx9IEgce0afeEMMg?usp=sharing)

The decision tree for diabetes classification that we learned in class relied heavily on the “Glycohemoglobin”
feature. In fact, this feature is the value of the specialized A1C blood test that is commonly used to determine
whether or not a patient has diabetes – it represents a sort of average blood sugar of the patient over time.
A1C works well for diagnosing whether or not a patient actually has diabetes if that disease is suspected.
However, what about before diabetes is suspected and that specialized test is ordered by a physician?
In this problem, you will try to determine a set of alternative features that could be used to diagnose diabetes.
The resulting decision tree could be useful, for example, for primary care physicians to screen for diabetes
or in locations without immediate access to the A1C blood test. We will first build two useful utility functions, and then use these functions to solve the problem.

* 2 - [Linear + Polynomial Regression](https://github.com/roynwu/Applied-Machine-Learning/blob/master/2.%20Linear%20%2B%20Polynomal%20Regression/notebook/linear%2Bpoly_regression.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1aIYae6nJXeQlOmgZYpI700C2fGZUmMIg?usp=sharing)

In this exercise, we will modify a linear regression implementation to fit a polynomial model
and explore the bias/variance tradeoff. 

* 3 - [Logistic Regression](https://github.com/roynwu/Applied-Machine-Learning/blob/master/3.%20Logistic%20Regression/notebook/logistic_regression.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1T4Fk8J47EXLyJ4bHCR9OQGndrX2uYsy2?usp=sharing)

Now that we’ve implemented a basic regression model using gradient descent, we will
use a similar technique to implement the logistic regression classifier.

* 4 - [AdaBoost](https://github.com/roynwu/Applied-Machine-Learning/blob/master/4.%20Adaboost/notebook/boosted.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1lJFy_A9Y69_vNTe1hubvhiUCNfcOL9U7?usp=sharing)

Boosted decision trees have been shown to be one of the best “out-of-the-box”
classifiers. (That is, if you know nothing about the data set and can’t do parameter tuning, they will likely
work quite well.) Boosting allows the decision trees to represent a much more complex decision surface than
a single decision tree. We write a class that implements a boosted decision tree classifier from scratch.

* 5 - [Deep Learning (CNNs)](https://github.com/roynwu/Applied-Machine-Learning/blob/master/5.%20Deep%20Learning%20(CNNs)/noteboook/cnn_classification.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1tPsoSiJMf33XWYHnFogkJk3H7WDS8G7T?usp=sharing)

Here, we will train a CNN to classify images of objects from a car racing video game, called SuperTuxKart.

* 6 - [Reinforcement Learning](https://github.com/roynwu/Applied-Machine-Learning/blob/master/6.%20Reinforcement%20Learning/notebook/mountain_car.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1iSQZtyBd7bSMSRamHups4tkLPawmCcOJ?usp=sharing)

Here, we will solve a classic problem in control, MountainCar, using two techniques
we learned in class: reinforcement learning and imitation learning.  

In MountainCar, the agent’s actions control a car that is initially located at the bottom of a valley between
two mountains, as seen in Fig 1. Success is defined as reaching the flag at the top of the hill on the
right. However, the car has limited engine power, so the task is not quite as easy as driving to the right.
Instead, a good policy would swing back and forth between the two hills for a while, slowly building up
enough momentum to eventually be able to get to the flag. This task was initially developed to stress-test
exploration (recall the “exploration-exploitation tradeoff”) in reinforcement learning algorithms: success
requires that the agent explore states far away from the target.  

We will use the OpenAI Gym version of the MountainCar environment, where the agent gets a negative
reward of -1 for every step spent in the environment without reaching the target. An episode ends when you
have spent 200 steps in the environment without reaching the flag (minimum reward = -200), or when you
reach the flag. Please read the python notebook introduction for more details on this environment.
This programming section is aimed at getting you acquainted with OpenAI gym, RL, and imitation learning.
