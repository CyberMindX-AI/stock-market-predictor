from automation import run_automation
if __name__ == "__main__":
    tickers = ["AAPL", "MSFT"] 
    for ticker in tickers:
        run_automation(ticker)