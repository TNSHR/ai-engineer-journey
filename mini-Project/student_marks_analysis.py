import pandas as pd
import matplotlib.pyplot as plt

#Load dataset
df = pd.read_csv("mini-Project/students.csv")

#View data
print("Dataset:")
print(df)

print("\nBasic Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

#Filter valid marks

df = df[df["marks"]>=0]

print("\nFiltered Dataset:")
print(df)

#Avetage Marks
average_marks = df["marks"].mean()
print(f"\nAverage Marks: {average_marks}")

#pass / Fail logic
df["result"] = df["marks"].apply(lambda x: "Pass" if x>= 40 else "Fail")

print("\nResults:")
print(df)

#Bar Chart
plt.bar(df["name"], df["marks"])
plt.axhline(y=40, color="red", linestyle="--", label="Pass Mark")
plt.xlabel("Students")
plt.ylabel("Marks")
plt.title("Student Marks Analysis")
plt.legend()
plt.show()
