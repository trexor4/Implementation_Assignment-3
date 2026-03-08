# CS325 Implementation Assignment 3 — Report

## Problem 1 — Minimize Maximum Absolute Deviation (Chebyshev Regression)

### LP Formulation (general)
We are given points \((x_i, y_i)\) for \(i=1\ldots n\). We want to find a line \(y = a x + b\) minimizing the maximum absolute deviation:

Objective:
\[ \text{minimize } E \]

Constraints (for each data point):
\[ a x_i + b - y_i \le E \]
\[ -(a x_i + b - y_i) \le E \]

Here, \(E\) is a variable representing the maximum absolute deviation.

### Specific instance (data points)
Points:
- (1, 3)
- (2, 5)
- (3, 7)
- (5, 11)
- (7, 14)
- (8, 15)
- (10, 19)

### Solver output (CBC via PuLP)
```
Optimal a: 1.7142857
Optimal b: 1.8571429
Optimal E (max abs deviation): 0.57142857
Check max abs deviation: 0.5714286000000008
```

The solver output above confirms an optimal solution was found (`Optimal objective value 0.57142857`).

### Plot
- The file `regression_plot.png` shows the data points and the fitted line.

---

## Problem 2 — Temperature Fit (Linear Trend + Seasonal + Solar Components)

### LP Formulation (general)
We model daily average temperature as:

\[ T(d) = x_0 + x_1 d
    + x_2 \cos\left(\frac{2\pi d}{365.25}\right) + x_3 \sin\left(\frac{2\pi d}{365.25}\right) \\
    + x_4 \cos\left(\frac{2\pi d}{365.25 \times 10.7}\right) + x_5 \sin\left(\frac{2\pi d}{365.25 \times 10.7}\right) \]

We find \(x_0, \dots, x_5\) minimizing the maximum absolute error:

Objective:
\[ \text{minimize } E \]

Constraints (for each record \(d_i, T_i\)):
\[ x_0 + x_1 d_i + x_2 c_{i,1} + x_3 s_{i,1} + x_4 c_{i,2} + x_5 s_{i,2} - T_i \le E \]
\[ -(x_0 + x_1 d_i + x_2 c_{i,1} + x_3 s_{i,1} + x_4 c_{i,2} + x_5 s_{i,2} - T_i) \le E \]

where
\(c_{i,1} = \cos\left(\frac{2\pi d_i}{365.25}\right)\),
\(s_{i,1} = \sin\left(\frac{2\pi d_i}{365.25}\right)\),
\(c_{i,2} = \cos\left(\frac{2\pi d_i}{365.25 \cdot 10.7}\right)\),
\(s_{i,2} = \sin\left(\frac{2\pi d_i}{365.25 \cdot 10.7}\right)\).

### Solver output (CBC via PuLP)
```
LP status: Optimal
Optimal solution found:
x0 = 8.0214197
x1 = 0.00010694836  (daily drift in °C)
x2 = 4.2808907
x3 = 8.1868578
x4 = -0.79063079
x5 = -0.29536021
Minimum maximum absolute deviation (E) = 14.23554
Estimated annual drift (x1 * 365.25) = 0.03906288849 °C/year
Estimated centennial drift (x1 * 365.25 * 100) = 3.906288849 °C/century
```

### Interpretation (warming/cooling trend)
- The estimated drift is **≈ 0.0391 °C/year**, which is **≈ 3.91 °C per century**.
- Since this is positive, the model predicts a **warming trend** for Corvallis.

### Plot
- The file `temperature_fit.png` shows:
  - raw data points (red)
  - the best-fit curve (blue)
  - the linear trend component \(x_0 + x_1 d\) (green)

---

## Running the Code

### Requirements
- Python 3
- Packages: `pulp`, `numpy`, `pandas`, `matplotlib`

Install dependencies with:
```bash
python -m pip install pulp numpy pandas matplotlib
```

### Execution
From the directory containing the files, run:
```bash
python problem1.py
python problem2.py
```

The scripts will print solver output and create the plot files `regression_plot.png` and `temperature_fit.png`.
