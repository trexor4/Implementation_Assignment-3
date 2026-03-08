import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pulp import LpProblem, LpMinimize, LpVariable, LpStatus, value

# Load and prepare the data
# The provided CSV file uses ';' as the delimiter.
data = pd.read_csv("Corvallis.csv", delimiter=";")

# We use the 'day' column (days since May 1, 1952) as d_i and the 'average' column as T_i.
d_values = data['day.1'].values
T_values = data['average'].values

# Define the periods for the two sinusoidal components.
P_season = 365.25              # period for the annual cycle
P_solar = 365.25 * 10.7        # period for the solar cycle


def chebyshev_temperature_fit(d_values, T_values, P_season, P_solar):
    """Fit a model using linear programming with Chebyshev (min-max) error.

    Model:
      T(d) = x0 + x1*d
             + x2*cos(2*pi*d/P_season) + x3*sin(2*pi*d/P_season)
             + x4*cos(2*pi*d/P_solar)  + x5*sin(2*pi*d/P_solar)

    Minimize E such that |T_i - T(d_i)| <= E for all data points.
    """

    prob = LpProblem("TemperatureFit", LpMinimize)

    x0 = LpVariable("x0", lowBound=None, upBound=None, cat="Continuous")
    x1 = LpVariable("x1", lowBound=None, upBound=None, cat="Continuous")
    x2 = LpVariable("x2", lowBound=None, upBound=None, cat="Continuous")
    x3 = LpVariable("x3", lowBound=None, upBound=None, cat="Continuous")
    x4 = LpVariable("x4", lowBound=None, upBound=None, cat="Continuous")
    x5 = LpVariable("x5", lowBound=None, upBound=None, cat="Continuous")
    E = LpVariable("E", lowBound=0, cat="Continuous")

    prob += E

    for d, T in zip(d_values, T_values):
        seasonal_cos = np.cos(2 * np.pi * d / P_season)
        seasonal_sin = np.sin(2 * np.pi * d / P_season)
        solar_cos = np.cos(2 * np.pi * d / P_solar)
        solar_sin = np.sin(2 * np.pi * d / P_solar)

        model_value = (
            x0
            + x1 * d
            + x2 * seasonal_cos
            + x3 * seasonal_sin
            + x4 * solar_cos
            + x5 * solar_sin
        )

        prob += model_value - T <= E
        prob += -(model_value - T) <= E

    prob.solve()

    if LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"LP did not reach optimal solution: {LpStatus[prob.status]}")

    return (
        value(x0),
        value(x1),
        value(x2),
        value(x3),
        value(x4),
        value(x5),
        value(E),
        LpStatus[prob.status],
    )


if __name__ == "__main__":
    x0, x1, x2, x3, x4, x5, E, status = chebyshev_temperature_fit(
        d_values, T_values, P_season, P_solar
    )

    print("LP status:", status)
    print("Optimal solution found:")
    print(f"x0 = {x0}")
    print(f"x1 = {x1}  (daily drift in °C)")
    print(f"x2 = {x2}")
    print(f"x3 = {x3}")
    print(f"x4 = {x4}")
    print(f"x5 = {x5}")
    print(f"Minimum maximum absolute deviation (E) = {E}")
    print(f"Estimated annual drift (x1 * 365.25) = {x1 * 365.25} °C/year")
    print(f"Estimated centennial drift (x1 * 365.25 * 100) = {x1 * 365.25 * 100} °C/century")

    # Plot the D and T values
    plt.figure(figsize=(14, 6))
    plt.plot(d_values, T_values, 'ro', label="Data points")
    plt.xlabel("Day")
    plt.ylabel("Temperature (°C)")

    # Plot the fitted model
    d_values_sorted = np.sort(d_values)
    T_model_values = [
        x0
        + x1 * d
        + x2 * np.cos(2 * np.pi * d / P_season)
        + x3 * np.sin(2 * np.pi * d / P_season)
        + x4 * np.cos(2 * np.pi * d / P_solar)
        + x5 * np.sin(2 * np.pi * d / P_solar)
        for d in d_values_sorted
    ]
    plt.plot(d_values_sorted, T_model_values, 'b-', label="Fitted model")

    # Plot the linear trend
    T_trend_values = [x0 + x1 * d for d in d_values_sorted]
    plt.plot(d_values_sorted, T_trend_values, 'g-', label="Linear trend")

    plt.title("Temperature Fit with Seasonal and Solar Components")
    plt.legend()
    plt.grid(True)
    plt.savefig("temperature_fit.png")
    plt.show()

    print("Plot saved as 'temperature_fit.png'")
