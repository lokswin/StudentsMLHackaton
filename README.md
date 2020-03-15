# StudentsMLHackaton

In this project we train our model for next purpose:
- Predict whether the student will be expelled before the end of the term of study or not.
- Get prediction accuracy [0.0 ... 1.0]

How does it work?

First block:
- Open train dataset csv file into buffer
- Read data from buffer
- Filter and convert some fields and rows
- Using Support Vector Machines metod train model (probability=True for predict_proba)
- Save model to file .pkl

Second block:
- Open test dataset csv file into buffer
- Read data from buffer
- Filter and convert some fields and rows
- Load our model file
- Generate prediction for test dataset
- Get prediction accuracy
- Print
