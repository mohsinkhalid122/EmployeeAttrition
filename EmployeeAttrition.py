#!/usr/bin/env python
# coding: utf-8

#load required packages
import pandas as pd
import numpy as np
import seaborn as sns
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier,GradientBoostingClassifier,RandomForestClassifier
from sklearn.metrics import confusion_matrix,roc_curve,auc,classification_report
from sklearn.svm import SVC
pd.set_option('display.max_columns', None)

#load required helper functions
from wrappers.helper_functions import (feature_selection,regularized_mean_encoding,
                                       plot_confusion_matrix,ABS_SHAP,Decile_Analysis,Eval_Statistics)


#load dataset
employee_data = pd.read_csv('employee-attrition.csv',sep='\t')

#binary encoding for categorical variables -- Gender, OverTime, Attrition (Target)
employee_data['Gender'] = employee_data['Gender'].replace({'Male': 1, 'Female': 0})
employee_data['OverTime'] = employee_data['OverTime'].replace({'Yes': 1, 'No': 0})
employee_data['Attrition'] = employee_data['Attrition'].replace({'Yes': 1, 'No': 0})

#one-hot encode BusinessTravel, Department, MaritalStatus
employee_data = pd.get_dummies(employee_data, columns=['BusinessTravel','Department','MaritalStatus'])

#drop excess variables for dimensionality reduction
employee_data.drop(['BusinessTravel_Non-Travel','Department_Human Resources','MaritalStatus_Divorced'],axis=1,inplace=True)

#drop uniformily distributed variables
employee_data.drop(['EmployeeNumber','EmployeeCount','Over18','StandardHours'],axis=1,inplace=True)

#append target at the end for clear visualization of the dataset
Target = employee_data.pop('Attrition')
employee_data.insert(len(employee_data.columns), 'Attrition', Target)

#split the dataset into training and testing datasets
train, test = train_test_split(employee_data,
                                   test_size=0.2,
                                   stratify=employee_data['Attrition'],
                                   random_state=42)

#apply regularized mean encoding on EducationField & JobRole
train,test = regularized_mean_encoding(train,test,'EducationField',1)
train,test = regularized_mean_encoding(train,test,'JobRole',1)

#drop coorealted variables
train.drop(['MonthlyIncome'],axis=1,inplace=True)
test.drop(['MonthlyIncome'],axis=1,inplace=True)

#Selected Variables
train = feature_selection(train, 27, 'Attrition')
test = test[list(train.columns.values)]

#Split target arrays
y_train = train['Attrition']
X_train = train.drop(['Attrition'],axis=1)
y_test = test['Attrition']
X_test = test.drop(['Attrition'],axis=1)

#Train Boosting Model
gb_model = GradientBoostingClassifier(n_estimators=50)
gb_model.fit(X_train,y_train)


#Define classes based on cut-off value
p_test = (gb_model.predict_proba(X_test)[:, 1] > 0.15).astype('float')

#find stats
cnf_matrix = confusion_matrix(y_test, p_test)
Eval_Statistics(cnf_matrix)




