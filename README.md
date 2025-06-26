# Vasicek-Interest-Rate-Model

Vasicek Interest Rate Model (Python)
This project implements the Vasicek short-rate model using historical 3-month Treasury Bill data (^IRX) from Yahoo Finance.

It includes:

Parameter estimation via Maximum Likelihood

Interest rate simulation using Monte Carlo

Visualization of simulated paths

 How it works
Data Download: Pulls historical T-bill rates using yfinance.

Model Calibration: Estimates a, b, and σ using historical rate changes.

Simulation: Generates interest rate paths under the Vasicek dynamics.

Plotting: Visualizes the first 50 of 10,000 simulated paths.

 Files
vasicek_model.py – main script

README.md – this file

**to run**


**pip install pandas numpy yfinance scipy matplotlib**


**python vasicek_model.py**


Output
Prints estimated Vasicek parameters

Shows a plot of interest rate simulations over 1 year (252 steps)

