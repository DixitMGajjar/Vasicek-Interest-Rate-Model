import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Step 1: Get data
def get_data(ticker, start, end):
    return yf.download(ticker, start=start, end=end)

# Step 2: Clean and preprocess data
def clean_data(raw_df):
    df = pd.DataFrame()
    df["actual_IRX"] = raw_df["Close"] / 100
    df["return"] = df["actual_IRX"].pct_change()
    df["daily_change"] = df["actual_IRX"].diff()
    df.dropna(inplace=True)
    return df

# Step 3: Define negative log likelihood for Vasicek
def negative_total_sum(params, df):
    a, b, sigma = params
    dt = 1/252
    r_t = df["actual_IRX"]
    r_lag = df["actual_IRX"].shift(1)
    delta_r = r_t - r_lag
    mu = a * (b - r_lag) * dt
    log_pdf = norm.logpdf(delta_r, loc=mu, scale=sigma)
    return -np.nansum(log_pdf)

# Step 4: Recalculate terms after calibration
def redo_calcu(df, a, b, sigma, dt=1/252):
    df["first_part_of_equ"] = a * (b - df["actual_IRX"].shift(1)) * dt
    df["actual_vs_calcu"] = df["daily_change"] - df["first_part_of_equ"]
    df["pdf"] = norm.pdf(df["actual_vs_calcu"], 0, sigma)
    df["ln_pdf"] = np.log(df["pdf"])
    return df

# Step 5: Simulate Vasicek paths
def calculate_int_rate(df, a, b, sigma, num_of_paths=10000):
    n_steps = 252
    dt = 1 / 252
    r0 = df["actual_IRX"].iloc[-1]
    paths = np.zeros((n_steps, num_of_paths))
    paths[0, :] = r0
    for t in range(1, n_steps):
        z = norm.ppf(np.random.rand(num_of_paths))
        paths[t] = (
            paths[t-1] +
            a * dt * (b - paths[t-1]) +
            sigma * np.sqrt(dt) * z
        )
    return paths

# Step 6: Plot Vasicek paths
def plot_fig(paths):
    plt.figure(figsize=(12, 6))
    plt.plot(paths[:, :50], alpha=0.5)
    plt.title("Vasicek Simulated Interest Rate Paths (50 of 10,000)")
    plt.xlabel("Days")
    plt.ylabel("Interest Rate")
    plt.grid(True)
    plt.show()

# ---- Main Execution ----
# Get and clean data
raw_data = get_data("^IRX", "2023-01-01", "2024-01-01")
df = clean_data(raw_data)

# Initial guess
initial_guess = [0.05, df["actual_IRX"].mean(), df["return"].std()]
result = minimize(negative_total_sum, initial_guess, args=(df,))
a, b, sigma = result.x
print("Estimated parameters:")
print(f"a: {a:.5f}, b: {b:.5f}, sigma: {sigma:.5f}")

# Optional: Recalculate components
df = redo_calcu(df, a, b, sigma)

# Simulate and plot
paths = calculate_int_rate(df, a, b, sigma)
plot_fig(paths)
