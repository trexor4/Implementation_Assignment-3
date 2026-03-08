import matplotlib.pyplot as plt
from pulp import LpProblem, LpMinimize, LpVariable, LpStatus, value

# Define the data points
points = [(1, 3), (2, 5), (3, 7), (5, 11), (7, 14), (8, 15), (10, 19)]

def chebyshev_regression(points):
    """Solve for the best-fit line y = a*x + b minimizing the maximum absolute error.

    This is also known as Chebyshev or minimax regression.

    Formulation:
        minimize E
        subject to: -E <= y_i - (a*x_i + b) <= E   for all points
    """

    prob = LpProblem("ChebyshevRegression", LpMinimize)

    a = LpVariable("a", lowBound=None, upBound=None, cat="Continuous")
    b = LpVariable("b", lowBound=None, upBound=None, cat="Continuous")
    E = LpVariable("E", lowBound=0, cat="Continuous")

    # Objective: minimize the maximum absolute deviation
    prob += E

    # Add constraints for each data point
    for (x, y) in points:
        prob += a * x + b - y <= E
        prob += -(a * x + b - y) <= E

    prob.solve()

    if LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"LP did not reach optimal solution: {LpStatus[prob.status]}")

    return value(a), value(b), value(E)


def max_abs_deviation(points, a, b):
    return max(abs(y - (a * x + b)) for x, y in points)


if __name__ == "__main__":
    a, b, E = chebyshev_regression(points)

    print("Optimal a:", a)
    print("Optimal b:", b)
    print("Optimal E (max abs deviation):", E)
    print("Check max abs deviation:", max_abs_deviation(points, a, b))

    # Plot the data points and the regression line
    xs = [x for x, _ in points]
    ys = [y for _, y in points]

    plt.figure()
    plt.plot(xs, ys, "ro", label="Data points")
    xs_line = [min(xs) - 1, max(xs) + 1]
    ys_line = [a * x + b for x in xs_line]
    plt.plot(xs_line, ys_line, "b-", label=f"Regression line (y={a:.3f}x + {b:.3f})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Chebyshev (min-max absolute error) regression")
    plt.legend()
    plt.grid(True)
    plt.savefig("regression_plot.png")
    plt.show()
    print("Plot saved as regression_plot.png")
