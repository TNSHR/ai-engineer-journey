import matplotlib.pyplot as plt

#Sample data

marks = [45,67,89,34,90]
students = ["A","B","C","D","E"]

#Bar Chart
plt.bar(students, marks)
plt.xlabel("Students")
plt.ylabel("Marks")
plt.title("Student Marks Analysis")

plt.show()