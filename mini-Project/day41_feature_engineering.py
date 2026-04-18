import pandas as pd

data = {
    "marks":[35,50,65,80,40,55,70,30,85,60],
    "attendance":[60,70,85,90,65,75,88,55,92,80],
    "study_hours":[1,2,4,6,2,3,5,1,7,4],
    "previous_score":[40,55,70,85,45,60,75,35,90,68],
    "at_risk":[1,0,0,0,1,0,0,1,0,0]
}

df = pd.DataFrame(data)
print("Original Dataset:")
print(df)

#Feature Engineering
df["effort_ratio"] = df["study_hours"]/df["attendance"]
df["performance"] = (df["marks"]+df["previous_score"])/2

print("\nDataset after feature Engineering:")
print(df)