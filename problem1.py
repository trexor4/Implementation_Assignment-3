
import matplotlib.pyplot as plt

# Define the data points
points = [(1, 3), (2, 5), (3, 7), (5, 11), (7, 14), (8, 15), (10, 19)]

### Code for linear programming ###

###################################

# display the results
print("Optimal a:", a)
print("Optimal b:", b)
print("Optimal t (max absolute deviation):", E)

# Plot the data points and the regression line
plt.figure()
plt.plot([x for (x, y) in points], [y for (x, y) in points], 'ro', label="Data points")
plt.plot([x for (x, y) in points], [a* x + b for (x, y) in points], 'b-', label="Regression line")
plt.legend()
plt.show()

# Save the plot
plt.savefig("regression_plot.png")
print("Plot saved as regression_plot.png")