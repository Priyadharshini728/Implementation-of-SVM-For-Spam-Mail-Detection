# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages.

2. Analyse the data.

3. Use modelselection and Countvectorizer to preditct the values.

4. Find the accuracy and display the result.


## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: PRIYADHARSHINI P
RegisterNumber:  212224040252
*/
```
~~~PYTHON
import pandas as pd
data=pd.read_csv("spam.csv", encoding='Windows-1252')
data

data.shape

x=data['v2'].values
y=data['v1'].values
x.shape

y.shape

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)
x_train

x_train.shape

~~~
~~~PYTHON
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc=accuracy_score(y_test,y_pred)
acc

con=confusion_matrix(y_test,y_pred)
print(con)

cl=classification_report(y_test,y_pred)
print(cl)

~~~
## Output:

# data

![image](https://github.com/user-attachments/assets/3b25c14d-e8bb-4176-86b2-7033f0c1e6f3)

# data.shape()

![image](https://github.com/user-attachments/assets/a4751df5-1d1a-45dd-ba5c-d7c355e1315c)


# x.shape()

![image](https://github.com/user-attachments/assets/7777a4a4-ae87-48e5-a316-c659bec61f47)


# y.shape()

![image](https://github.com/user-attachments/assets/98a2d04f-3fe9-4755-91e3-5da716ca4ece)


# x_train

![image](https://github.com/user-attachments/assets/fb5a47cc-29ab-4460-bb42-212fe723c16a)


# x_train.shape()

![image](https://github.com/user-attachments/assets/55ad649b-1045-437c-a61c-b028f391dff7)


# y_pred

![image](https://github.com/user-attachments/assets/7a832a0f-8f4a-48b1-9622-9f2dc741ee3f)


# acc (accuracy)

![image](https://github.com/user-attachments/assets/269b160b-e5de-4fdd-9a1e-b7a00668cd02)


# con (confusion matrix)

![image](https://github.com/user-attachments/assets/ab4cdd09-e257-401c-b075-20907069d921)


# cl (classification report)

![image](https://github.com/user-attachments/assets/7280d71a-8ba7-4076-8e7a-f00e8af4f3d2)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
