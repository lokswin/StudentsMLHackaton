#################################################################
#	This block take "test.csv" dataset and						#
#	print our prediction [0.0 ... 1.0]							#
#################################################################

import csv
import joblib
import numpy as np
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier

def ft_fill_zero(row):
  res = []
  for i in row:
    if i == "":
      i = "0"
    res.append(i)
  return res

def ft_gender(gender):
  if gender == "female" :
    return '0'
  else:
    return '1'

def ft_wordsum(charstr):
  tmp = sum(ord(ch) for ch in charstr)
  return tmp

results = []
with open("test.csv") as csvfile:
    reader = csv.reader(csvfile, delimiter=',') # change contents to floats
    for row in reader: # each row is a list
            row = ft_fill_zero(row)#replace empty field by 0
            row[3] = ft_gender(row[3])#convert each gender to 1(male) or 0
            row[1] = ft_wordsum(row[1])#convert each birth date field to number
            row[2] = ft_wordsum(row[2])#convert each city field to number
            row[6] = ft_wordsum(row[6])#convert each google_ads field to number
            row[7] = ft_wordsum(row[7])#convert each status  field to number
            row[24] = ft_wordsum(row[24])#convert each bsq field to number
            row[32] = ft_wordsum(row[32])#convert each date field to number
            del row[0]#delete id field
            results.append(row)
del results[0]#del header row
# LOAD model using joblib #
CLFmodel = joblib.load("modelFinal.pkl")
#model predicts a possible solution [0 - 1]
prediction = CLFmodel.predict(results)
#get accuracy [0 - 1] -> [0.0 ... 1.0]
churn_prob = CLFmodel.predict_proba(results)
#print [0.0 ... 1.0]
submission_contaract_field = churn_prob[:,1]
for i in submission_contaract_field:
  print(i)
