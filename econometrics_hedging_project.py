# Econometrics Project:
# Using Linear Regression to Assess Correlation in the Stock Market
# Target ticker: QQQ
# Test tickers: XLP, VOO
# Sample portfolio: Equal-weighted XLP + VOO


# -------------------------------
# 1. IMPORT LIBRARIES
# -------------------------------
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt

# -------------------------------
# 2. DEFINE TICKERS AND TIMEFRAME
# -------------------------------
target_ticker = "QQQ"
test_tickers = ["XLP", "VOO"]

start_date = "2020-01-01"
end_date = "2026-01-01"

all_tickers = [target_ticker] + test_tickers

# -------------------------------
# 3. DOWNLOAD ADJUSTED CLOSE DATA
# -------------------------------
data = yf.download(all_tickers, start=start_date, end=end_date, auto_adjust=True)["Close"]

# Drop missing values just in case
data = data.dropna()

print("Price Data Preview:")
print(data.head())

# -------------------------------
# 4. CALCULATE DAILY RETURNS
# -------------------------------
returns = data.pct_change().dropna()

print("\nDaily Returns Preview:")
print(returns.head())

# -------------------------------
# 5. CORRELATION MATRIX
# -------------------------------
corr_matrix = returns.corr()
  
print("\nCorrelation Matrix:")
print(corr_matrix)

# -------------------------------
# 6. SIMPLE LINEAR REGRESSION:
#    QQQ vs each test ticker individually
# -------------------------------
for ticker in test_tickers:
    print(f"\n{'='*60}")
    print(f"Simple Linear Regression: {target_ticker} ~ {ticker}")
    print(f"{'='*60}")

    X = returns[[ticker]]
    y = returns[target_ticker]

    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    print(model.summary())

# -------------------------------
# 7. MULTIPLE REGRESSION:
#    QQQ vs XLP and VOO together
# -------------------------------
print(f"\n{'='*60}")
print("Multiple Regression: QQQ ~ XLP + VOO")
print(f"{'='*60}")

X_multi = returns[test_tickers]
y_multi = returns[target_ticker]

X_multi = sm.add_constant(X_multi)
multi_model = sm.OLS(y_multi, X_multi).fit()

print(multi_model.summary())

# -------------------------------
# 8. BUILD A SAMPLE PORTFOLIO
#    Equal-weight portfolio of XLP and VOO
# -------------------------------
weights = np.array([0.5, 0.5])

returns["Sample_Portfolio"] = returns[test_tickers].dot(weights)

print("\nSample Portfolio Returns Preview:")
print(returns[["Sample_Portfolio"]].head())

# -------------------------------
# 9. REGRESSION:
#    QQQ vs Sample Portfolio
# -------------------------------
print(f"\n{'='*60}")
print("Regression: QQQ ~ Sample_Portfolio")
print(f"{'='*60}")

X_port = returns[["Sample_Portfolio"]]
y_port = returns[target_ticker]

X_port = sm.add_constant(X_port)
portfolio_model = sm.OLS(y_port, X_port).fit()

print(portfolio_model.summary())

# -------------------------------
# 10. HEDGE RATIO INTERPRETATION
# -------------------------------
beta_portfolio = portfolio_model.params["Sample_Portfolio"]
print(f"\nEstimated hedge ratio using Sample Portfolio: {beta_portfolio:.4f}")

# Example interpretation
print(
    f"This suggests that for each $1 exposure in {target_ticker}, "
    f"approximately ${beta_portfolio:.4f} of the sample portfolio "
    f"(50% XLP, 50% VOO) may be used as a hedge, based on historical linear sensitivity."
)

# -------------------------------
# 11. PLOT QQQ VS SAMPLE PORTFOLIO RETURNS
# -------------------------------
plt.figure(figsize=(8, 6))
plt.scatter(returns["Sample_Portfolio"], returns[target_ticker], alpha=0.5)
plt.xlabel("Sample Portfolio Daily Returns")
plt.ylabel("QQQ Daily Returns")
plt.title("QQQ vs Sample Portfolio Returns")

# Add regression line
x_vals = returns["Sample_Portfolio"]
y_vals = portfolio_model.predict(X_port)

sorted_idx = np.argsort(x_vals)
plt.plot(x_vals.iloc[sorted_idx], y_vals.iloc[sorted_idx])

plt.show()

# -------------------------------
# 12. COMPARE CUMULATIVE RETURNS
# -------------------------------
cum_returns = (1 + returns[[target_ticker, "XLP", "VOO", "Sample_Portfolio"]]).cumprod()

plt.figure(figsize=(10, 6))
for col in cum_returns.columns:
    plt.plot(cum_returns.index, cum_returns[col], label=col)

plt.title("Cumulative Growth of $1 Investment")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.legend()
plt.show()

# -------------------------------
# 13. SUMMARY STATISTICS TABLE
# -------------------------------
summary_stats = pd.DataFrame({
    "Mean Daily Return": returns[[target_ticker, "XLP", "VOO", "Sample_Portfolio"]].mean(),
    "Std Dev": returns[[target_ticker, "XLP", "VOO", "Sample_Portfolio"]].std(),
    "Min Return": returns[[target_ticker, "XLP", "VOO", "Sample_Portfolio"]].min(),
    "Max Return": returns[[target_ticker, "XLP", "VOO", "Sample_Portfolio"]].max()
})

print("\nSummary Statistics:")
print(summary_stats)

# -------------------------------
# 14. OPTIONAL: EXPORT RESULTS
# -------------------------------
summary_stats.to_csv("summary_statistics.csv")
corr_matrix.to_csv("correlation_matrix.csv")

print("\nFiles saved:")
print("- summary_statistics.csv")
print("- correlation_matrix.csv")
