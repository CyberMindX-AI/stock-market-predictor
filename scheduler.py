from apscheduler.schedulers.blocking import BlockingScheduler
from predict import run_automation

def scheduled_job():
    tickers = ["AAPL", "MSFT"]
    for ticker in tickers:
        run_automation(ticker)

if __name__ == "__main__":
    scheduler = BlockingScheduler()
    scheduler.add_job(scheduled_job, 'cron', hour=6)  # every day 6 AM
    print("Scheduler started. Waiting for the next run...")
    scheduler.start()
