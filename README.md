# Vasicek Interest Rate Model

Simple Python implementation for calibrating and simulating the Vasicek short-rate model using historical data.

## What it does

Downloads interest rate data, fits the Vasicek model parameters, and runs Monte Carlo simulations to forecast rate paths. The model assumes rates follow:

```
dr = a(b - r)dt + σ dW
```

Where:
- `a` = mean reversion speed
- `b` = long-term average rate  
- `σ` = volatility

## Setup

You'll need these packages:
```bash
pip install pandas numpy yfinance scipy matplotlib
```

## Basic usage

```python
# Get data and calibrate
data = dwnld_cln_data("^IRX", "2023-01-01", "2024-12-01")
a, b, sigma = calibrate_vasicek(data)

# Run simulation
r0 = data["actual_IRX"].iloc[-1]
paths = simulate_vasicek_paths(r0, a, b, sigma, T=1, n_steps=252, n_paths=5000)

# Plot results
plot_paths(paths, n_plot=50)
```

## What the parameters mean

- **a (0.1-2.0)**: How fast rates snap back to the long-term average. Higher = faster mean reversion
- **b (0.02-0.08)**: Where rates settle in the long run. Usually matches historical average
- **σ (0.01-0.3)**: How much randomness/volatility in day-to-day moves

## Things to know

The Vasicek model can produce negative rates, which might or might not be realistic depending on your use case. It's simple and fast but assumes constant volatility.

Works best with:
- At least 6 months of daily data
- Periods without major regime changes
- Short to medium-term forecasting (1-3 years)

The calibration uses maximum likelihood estimation and should converge automatically. If you get weird parameters, try a different date range or check your data quality.

## File structure

Main functions:
- `dwnld_cln_data()` - gets and cleans Yahoo Finance data
- `calibrate_vasicek()` - estimates model parameters  
- `simulate_vasicek_paths()` - runs Monte Carlo simulation
- `plot_paths()` - makes charts of the results

That's pretty much it. The code is straightforward and you can easily modify the date ranges, simulation parameters, or add your own data sources.
