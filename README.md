# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING 

 import pandas as pd
 import numpy as np
 import seaborn as sns

 from sklearn.model_selection import train_test_split
 from sklearn.neighbors import KNeighborsClassifier
 from sklearn.metrics import accuracy_score, confusion_matrix

 data=pd.read_csv("income(1) (1).csv",na_values=[ " ?"])
 data


# OUTPUT:


<img width="1358" height="790" alt="image" src="https://github.com/user-attachments/assets/3e1d69bd-f582-4280-8481-3a6cab014805" />


```
 data.isnull().sum()

```

# Output:


<img width="259" height="655" alt="image" src="https://github.com/user-attachments/assets/e029ef9d-1484-43b5-b328-f58abad33067" />


```
 missing=data[data.isnull().any(axis=1)]
 missing

```

# Output:


<img width="1391" height="846" alt="image" src="https://github.com/user-attachments/assets/671e4fb0-9097-48a0-b14b-b09350791a9c" />


```
 data2=data.dropna(axis=0)
 data2

```

# Output:


<img width="1397" height="846" alt="image" src="https://github.com/user-attachments/assets/2e52b61e-5b5d-4009-b5a3-a664ff490eec" />


```
 sal=data["SalStat"]

 data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
 print(data2['SalStat'])

```

# Output:


<img width="1240" height="335" alt="image" src="https://github.com/user-attachments/assets/8ab791c8-8ad6-406d-966c-f458db23e67b" />


```
  sal2=data2['SalStat']

  dfs=pd.concat([sal,sal2],axis=1)
  dfs

```

# Output:


<img width="520" height="596" alt="image" src="https://github.com/user-attachments/assets/ae6520d7-90f1-4734-85a0-6b34fce239be" />


```
 data2

```

# Output:


<img width="1403" height="763" alt="image" src="https://github.com/user-attachments/assets/baa42dd2-4058-4902-b853-427a5c02d0dc" />


```
 new_data=pd.get_dummies(data2, drop_first=True)
 new_data

```

# Output:


<img width="1403" height="666" alt="image" src="https://github.com/user-attachments/assets/8e6b5bba-92ae-4a6a-b7ae-1f15d87a3fa6" />


```
 columns_list=list(new_data.columns)
 print(columns_list)

```

# Output:


<img width="1390" height="136" alt="image" src="https://github.com/user-attachments/assets/6c4ecae5-813d-4664-8335-3c4e7bcf5881" />


```
 features=list(set(columns_list)-set(['SalStat']))
 print(features)

```
# Output:


<img width="1392" height="124" alt="image" src="https://github.com/user-attachments/assets/3928a16b-0270-4204-aa76-198c32a0de03" />


```
 y=new_data['SalStat'].values
 print(y)

```
# Output:


<img width="356" height="111" alt="image" src="https://github.com/user-attachments/assets/e3b3e6d2-2b15-4533-ba17-672668cff74e" />


```
 x=new_data[features].values
 print(x)

```

# Output:


<img width="530" height="243" alt="image" src="https://github.com/user-attachments/assets/5ffe9258-27a4-4674-8eb1-d5a9b885ccda" />


```
 train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
  KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
  KNN_classifier.fit(train_x,train_y)

```

# Output:


<img width="576" height="169" alt="image" src="https://github.com/user-attachments/assets/ddc5dbf8-0373-4931-98fa-fc112deb174c" />


```
  prediction=KNN_classifier.predict(test_x)

  confusionMatrix=confusion_matrix(test_y, prediction)
  print(confusionMatrix)

```

# Output:


<img width="585" height="131" alt="image" src="https://github.com/user-attachments/assets/92da8694-925b-4316-90de-e8455cfe2a2a" />


```
 accuracy_score=accuracy_score(test_y,prediction)
 print(accuracy_score)

```
# OUtput:


<img width="557" height="109" alt="image" src="https://github.com/user-attachments/assets/0238f118-81e4-4204-92c9-1886c5c682e8" />


```
 print("Misclassified Samples : %d" % (test_y !=prediction).sum())

```

# Output:


<img width="702" height="84" alt="image" src="https://github.com/user-attachments/assets/f3325be8-0247-4e4e-82de-0955fbb3b8fe" />


```
 data.shape

```

# Output:


<img width="191" height="90" alt="image" src="https://github.com/user-attachments/assets/433bace2-dd58-47e7-b9b9-ab0d7d3fb751" />


```
 import pandas as pd
 from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif

data={
 'Feature1': [1,2,3,4,5],
 'Feature2': ['A','B','C','A','B'],
 'Feature3': [0,1,1,0,1],
 'Target'  : [0,1,1,0,1]
 }

 df=pd.DataFrame(data)
 x=df[['Feature1','Feature3']]
 y=df[['Target']]

 selector=SelectKBest(score_func=mutual_info_classif,k=1)
 x_new=selector.fit_transform(x,y)

```
# Output:


<img width="1392" height="153" alt="image" src="https://github.com/user-attachments/assets/218cb1a2-9a95-4089-891f-b347b28f128e" />


```
selected_feature_indices=selector.get_support(indices=True)

```


# RESULT:
        Thus the program to read the given data and perform Feature Scaling and Feature Selection process and
 save the data to a file is been executed.
