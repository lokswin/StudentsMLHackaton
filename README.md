# StudentsMLHackaton

In this project we train our model for next purpose:
- Predict whether the student will be expelled before the end of the term of study or not.
- Get prediction accuracy [0.0 ... 1.0]

How does it work?
Frist block
- Open csv file into buffer
- Read data from buffer
- Filter fields and rows
- Using Support Vector Machines metod train model (probability=True for predict_proba)
- Save model to file .pkl
