#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing packages
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# In[5]:


# Pull in Data Set
data = {

    'Age': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100],

    'BloodPressure': [120, 122, 126, 128, 130, 133, 135, 138, 142, 145, 150, 155, 160, 165, 170, 175]

}


# In[6]:


df = pd.DataFrame(data)

df_descriptive = df.describe()

print(df_descriptive)


# In[9]:


# scatter plot of age vs blood pressure
plt.figure(figsize=(8,6))
plt.scatter(df['Age'], df ['BloodPressure'], color='blue')
plt.xlabel('Age')
plt.ylabel('BloodPressure')
plt.title("Age vs. Blood Pressure")
plt.show


# In[8]:


# Linear regression model
X = df[['Age']]
y = df['BloodPressure']
regression = LinearRegression().fit(X, y)


# In[24]:


plt.plot(X, regression.predict(X)), label = "Regression Line", color = "red"
plt.scatter(df['Age'], df['BloodPressure'], color='blue')
plt.show


# In[23]:


slope = regression.coef_[0]
intercept = regression.intercept_
print(f"Regression model has slope of {slope:.2f} and intercepts of {intercept:.2f}.")


# In[17]:


#Predictions
new_ages = [30, 40, 50, 60]
df_ages = pd.DataFrame({'Age' : new_ages})
predicted_blood_pressures = regression.predict(df_ages)


# In[21]:


for age, bp in zip(new_ages, predicted_blood_pressures):
    print(f"Predicted Blood Pressure at Age {age} is {bp:.2f}.")


# In[ ]:




