import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#Dataset
data = {
    "marks":[35,50,65,80,40,55,70,30,85,60],
    "attendance":[60,70,85,90,65,75,88,55,92,80]
}

df = pd.DataFrame(data)
print("Original Data:")
print(df)

#Standard Scaling
scaler = StandardScaler()
standard_scaled = scaler.fit_transform(df)

print("\nStandard Scaled Data:")
print(standard_scaled)
#MinMax Scaling
minmax = MinMaxScaler()
minmax_scaled = minmax.fit_transform(df)

print("\nMinMax Scaled Data:")
print(minmax_scaled)