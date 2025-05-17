# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import dataset and print head,info of the dataset

2.check for null values

3.Import kmeans and fit it to the dataset

4.Plot the graph using elbow method

5.Print the predicted array

6.Plot the customer segments

## Program:

/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Jayaseelan U
RegisterNumber:  212223220039
*/


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv("Mall_Customers.csv")

print(data.head())
print(data.info())
print(data.isnull().sum())

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
    kmeans.fit(data.iloc[:, 3:5])
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.grid(True)
plt.show()

km = KMeans(n_clusters=5, init="k-means++", random_state=42)
y_pred = km.fit_predict(data.iloc[:, 3:5])
data["Cluster"] = y_pred

plt.figure(figsize=(8, 6))
colors = ['red', 'black', 'blue', 'green', 'magenta']
for i in range(5):
    cluster = data[data["Cluster"] == i]
    plt.scatter(cluster["Annual Income (k$)"], cluster["Spending Score (1-100)"], 
                c=colors[i], label=f"Cluster {i}")

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segments")
plt.legend()
plt.grid(True)
plt.show()


## Output:
## 1.DATA.HEAD():
![Screenshot 2025-05-10 133305](https://github.com/user-attachments/assets/a2394bb4-de89-4c65-af62-58b6110afa9c)
## 2.DATA.INF0():
![Screenshot 2025-05-10 133313](https://github.com/user-attachments/assets/58b0a731-72e5-426f-8255-19c2ccc4626b)
## 3.DATA.ISNULL().SUM():
![Screenshot 2025-05-10 133321](https://github.com/user-attachments/assets/5d8cc726-b6dd-4025-9f69-094ec165f021)
## 4.PLOT USING ELBOW METHOD:
![output](https://github.com/user-attachments/assets/e243fa12-65a0-419f-8078-b5659a2a0e5d)
## 5.Y_PRED ARRAY:
![Screenshot 2025-05-10 133239](https://github.com/user-attachments/assets/f0b66a23-4b94-4b13-af29-3fa5301fd6f7)
## 6.CUSTOMER SEGMENT:
![output2](https://github.com/user-attachments/assets/2e954c20-9caf-49a9-bf5d-f04503131cff)
## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
