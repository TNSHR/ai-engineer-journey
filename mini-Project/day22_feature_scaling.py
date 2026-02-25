import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

data = {
    "experience_years": [1,2,3,4,5],
    "salary":[3,4,6,8,10]
}

df = pd.DataFrame(data)
print("Before Scaling:\n", df)

# Features and target variable
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

scaled_df = pd.DataFrame(scaled_data, columns=df.columns)

print("After Standard Scaling:\n", scaled_df)