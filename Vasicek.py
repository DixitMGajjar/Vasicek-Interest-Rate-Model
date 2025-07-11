import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def dwnld_cln_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    df = pd.DataFrame()
    df["actual_IRX"] = data["Close"] / 100  # Cnvrt % to .
    df["return"] = df["actual_IRX"].pct_change()
    df["daily_change"] = df["actual_IRX"].diff()
    df.dropna(inplace=True)
    return df


def calibrate_vasicek(df, n_days=252):

    dt = 1 / n_days
    
    
    rates = df["actual_IRX"]
    a_init = 0.1  
    b_init = rates.mean()
    sigma_init = rates.std() * np.sqrt(n_days)  
    
    def negative_log_likelihood(params):
        a, b, sigma = params
        
        if a <= 0 or sigma <= 0:
            return np.inf
            
        r_t = df["actual_IRX"].values
        r_lag = np.roll(r_t, 1)[1:]
        r_t = r_t[1:]  
        
        mu = a * (b - r_lag) * dt
        residuals = r_t - r_lag - mu
        log_likelihood = -0.5 * len(r_t) * np.log(2 * np.pi * sigma**2 * dt) - \
                        np.sum(residuals**2) / (2 * sigma**2 * dt)
        
        return -log_likelihood
    
    
    bounds = [(0.001, 5.0),
              (0.0, 0.2),  
              (0.001, 1.0)]
    
    result = minimize(negative_log_likelihood, 
                     [a_init, b_init, sigma_init], 
                     bounds=bounds,
                     method='L-BFGS-B')
    
    if not result.success:
        print(f"Warning: Opt may not have converged. Message: {result.message}")
    
    a, b, sigma = result.x
    return a, b, sigma


def simulate_vasicek_paths(r0, a, b, sigma, T, n_steps, n_paths):
    
    dt = T / n_steps
    rates = np.zeros((n_steps + 1, n_paths))
    rates[0] = r0
    
    
    Z = np.random.normal(size=(n_steps, n_paths))
    
    for t in range(1, n_steps + 1):
        drift = a * (b - rates[t-1]) * dt
        diffusion = sigma * np.sqrt(dt) * Z[t-1]
        rates[t] = rates[t-1] + drift + diffusion
        
    
    
    return rates


def calculate_statistics(paths):
    
    final_rates = paths[-1]
    
    stats = {
        'mean_final_rate': np.mean(final_rates),
        'std_final_rate': np.std(final_rates),
        'min_final_rate': np.min(final_rates),
        'max_final_rate': np.max(final_rates),
        'percentile_5': np.percentile(final_rates, 5),
        'percentile_95': np.percentile(final_rates, 95)
    }
    
    return stats


def plot_paths(paths, n_plot=50, T=1):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    
    time_steps = np.linspace(0, T, paths.shape[0])
    ax1.plot(time_steps, paths[:, :n_plot], lw=0.6, alpha=0.7)
    ax1.set_title(f"Sample Vasicek Interest Rate Paths (n={n_plot})")
    ax1.set_xlabel("Time (Years)")
    ax1.set_ylabel("Interest Rate")
    ax1.grid(True, alpha=0.3)
    
    
    final_rates = paths[-1]
    ax2.hist(final_rates, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(np.mean(final_rates), color='red', linestyle='--', 
                label=f'Mean: {np.mean(final_rates):.4f}')
    ax2.set_title("Distribution of Final Interest Rates")
    ax2.set_xlabel("Interest Rate")
    ax2.set_ylabel("Density")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def analyze_model_fit(df, a, b, sigma, n_days=252):
    
    dt = 1 / n_days
    rates = df["actual_IRX"].values
    
    # Calculate theoretical vs actual statistics
    theoretical_mean = b
    theoretical_variance = sigma**2 / (2 * a)
    
    actual_mean = np.mean(rates)
    actual_variance = np.var(rates)
    
    print("=== MODEL CALIBRATION RESULTS ===")
    print(f"Mean reversion speed (a): {a:.4f}")
    print(f"Long-term mean (b): {b:.4f} ({b*100:.2f}%)")
    print(f"Volatility (σ): {sigma:.4f}")
    print(f"\n=== MODEL FIT ANALYSIS ===")
    print(f"Theoretical long-term mean: {theoretical_mean:.4f} ({theoretical_mean*100:.2f}%)")
    print(f"Actual mean: {actual_mean:.4f} ({actual_mean*100:.2f}%)")
    print(f"Theoretical variance: {theoretical_variance:.6f}")
    print(f"Actual variance: {actual_variance:.6f}")


###################### MAIN ########################

if __name__ == "__main__":
    # Step 1: Download and clean data
    # You can adjust these dates for any time period
    start_date = "2020-01-01"  # Can be any start date
    end_date = "2024-12-01"    # Can be any end date
    
    print("Downloading data...")
    data = dwnld_cln_data("^IRX", start_date, end_date)
    print(f"Data period: {start_date} to {end_date}")
    print(f"Number of observations: {len(data)}")
    
    # Step 2: Calibrate Vasicek parameters
    print("\nCalibrating Vasicek model...")
    a, b, sigma = calibrate_vasicek(data)
    
    # Analyze model fit
    analyze_model_fit(data, a, b, sigma)
    
    # Step 3: Simulate multiple paths
    r0 = data["actual_IRX"].iloc[-1]  # Use last observed rate as starting point
    T = 1  # Time horizon in years (can be adjusted)
    n_steps = 252  # Daily steps for 1 year
    n_paths = 10000
    
    print(f"\nSimulating {n_paths} paths over {T} year(s)...")
    np.random.seed(42)  # For reproducibility
    paths = simulate_vasicek_paths(r0, a, b, sigma, T, n_steps, n_paths)
    
    # Step 4: Calculate and display statistics
    stats = calculate_statistics(paths)
    print("\n=== SIMULATION STATISTICS ===")
    for key, value in stats.items():
        print(f"{key}: {value:.4f} ({value*100:.2f}%)")
    
    # Step 5: Plot results
    plot_paths(paths, n_plot=100, T=T)
