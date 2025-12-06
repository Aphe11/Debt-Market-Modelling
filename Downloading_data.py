# ================================================================
# Impact of COVID-19 on Corporate Dividend Policy
# ================================================================

# Required libraries
import pandas as pd
import yfinance as yf
import datetime
from pandas_datareader import data as pdr

# ================================================================
# 1. Set up parameters
# ================================================================
start_date = "2018-01-01"
end_date = "2022-12-31"

# Treatment group: COVID-affected industries (airlines, hospitality, retail)
treatment_firms = {
    "AAL": "Airline",      # American Airlines
    "DAL": "Airline",      # Delta Airlines
    "LUV": "Airline",      # Southwest Airlines
    "UAL": "Airline",      # United Airlines
    "MAR": "Hospitality",  # Marriott
    "HLT": "Hospitality",  # Hilton
    "CCL": "Cruise",       # Carnival
    "NCLH": "Cruise",      # Norwegian Cruise Line
    "MCD": "Restaurant",   # McDonald's
    "SBUX": "Restaurant"   # Starbucks
}

# Control group: Less-affected industries (tech, utilities, consumer staples)
control_firms = {
    "MSFT": "Tech",
    "AAPL": "Tech",
    "GOOGL": "Tech",
    "AMZN": "Tech",
    "NEE": "Utility",
    "DUK": "Utility",
    "PG": "ConsumerStaple",
    "KO": "ConsumerStaple",
    "PEP": "ConsumerStaple",
    "JNJ": "Healthcare"
}

# Combine both groups
all_firms = {**treatment_firms, **control_firms}

# ================================================================
# 2. Download firm-level data from Yahoo Finance
# ================================================================

def get_firm_data(ticker, industry):
    """Download stock price and dividend data for a given ticker"""
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date, actions=True)
    df = df.reset_index()
    df["Ticker"] = ticker
    df["Industry"] = industry
    return df

firm_data_list = []
for ticker, industry in all_firms.items():
    print(f"Downloading data for {ticker}...")
    try:
        firm_df = get_firm_data(ticker, industry)
        firm_data_list.append(firm_df)
    except Exception as e:
        print(f"Error downloading {ticker}: {e}")

firm_data = pd.concat(firm_data_list, ignore_index=True)
firm_data.to_csv("firm_data_raw.csv", index=False)
print(f"\nRaw firm data saved. Total rows: {len(firm_data)}")

# ================================================================
# 3. Clean and prepare firm data
# ================================================================
# Keep relevant columns
firm_data = firm_data[["Date", "Ticker", "Industry", "Close", "Dividends"]]

# Add Year and Quarter for merging with macro data
firm_data["Year"] = pd.to_datetime(firm_data["Date"]).dt.year
firm_data["Quarter"] = pd.to_datetime(firm_data["Date"]).dt.quarter

# Aggregate quarterly average price and total dividends
firm_quarterly = firm_data.groupby(["Ticker", "Industry", "Year", "Quarter"]).agg(
    Avg_Price=("Close", "mean"),
    Total_Dividends=("Dividends", "sum")
).reset_index()

print(f"Quarterly firm data prepared. Total rows: {len(firm_quarterly)}")

# ================================================================
# 4. Download macroeconomic data from FRED
# ================================================================
print("\nDownloading macroeconomic data from FRED...")
fred_codes = ["FEDFUNDS", "UNRATE", "INDPRO", "CPIAUCSL"]
start = datetime.datetime(2018, 1, 1)
end = datetime.datetime(2022, 12, 31)

try:
    macro_data = pdr.DataReader(fred_codes, "fred", start, end)
    macro_data = macro_data.resample("Q").mean().reset_index()  # convert to quarterly
    macro_data["Year"] = macro_data["DATE"].dt.year
    macro_data["Quarter"] = macro_data["DATE"].dt.quarter
    macro_data = macro_data.drop(columns=["DATE"])
    print("Macroeconomic data downloaded successfully")
except Exception as e:
    print(f"Warning: Could not download FRED data: {e}")
    print("Creating empty macro dataframe...")
    macro_data = pd.DataFrame(columns=["Year", "Quarter", "FEDFUNDS", "UNRATE", "INDPRO", "CPIAUCSL"])

# ================================================================
# 5. Merge firm and macro data
# ================================================================
merged_df = pd.merge(firm_quarterly, macro_data, on=["Year", "Quarter"], how="left")

# Add treatment and post indicators
merged_df["Treatment"] = merged_df["Industry"].isin(["Airline", "Hospitality", "Cruise", "Restaurant"]).astype(int)
merged_df["Post"] = ((merged_df["Year"] >= 2020) & (merged_df["Quarter"] >= 1)).astype(int)
merged_df["Treat_Post"] = merged_df["Treatment"] * merged_df["Post"]

# ================================================================
# 6. Save final dataset
# ================================================================
merged_df.to_csv("covid_dividend_macro_dataset.csv", index=False)
print("\nFinal dataset saved as covid_dividend_macro_dataset.csv")

# ================================================================
# 7. Display summary
# ================================================================
print("\n" + "="*60)
print("DATASET SUMMARY")
print("="*60)
print(f"\nTotal observations: {len(merged_df)}")
print(f"Firms included: {merged_df['Ticker'].nunique()}")
print(f"Time period: {merged_df['Year'].min()} - {merged_df['Year'].max()}")
print(f"\nTreatment firms: {merged_df[merged_df['Treatment']==1]['Ticker'].nunique()}")
print(f"Control firms: {merged_df[merged_df['Treatment']==0]['Ticker'].nunique()}")
print(f"\nPre-COVID observations: {len(merged_df[merged_df['Post']==0])}")
print(f"Post-COVID observations: {len(merged_df[merged_df['Post']==1])}")

print("\n" + "="*60)
print("FIRST FEW ROWS:")
print("="*60)
print(merged_df.head(10))

print("\n" + "="*60)
print("DIVIDEND SUMMARY BY GROUP:")
print("="*60)
dividend_summary = merged_df.groupby(["Treatment", "Post"]).agg({
    "Total_Dividends": ["mean", "sum", "count"]
}).round(4)
print(dividend_summary)

print("\n" + "="*60)
print("MISSING VALUES:")
print("="*60)
print(merged_df.isnull().sum())

# ================================================================
# 8. Add firm-level control variables
# ================================================================
print("\nAdding firm-level control variables from Yahoo Finance...")

firm_controls = []

for ticker in all_firms.keys():
    try:
        info = yf.Ticker(ticker).info
        
        firm_controls.append({
            "Ticker": ticker,
            "MarketCap": info.get("marketCap"),
            "PERatio": info.get("trailingPE"),
            "PB_Ratio": info.get("priceToBook"),
            "Beta": info.get("beta"),
            "DebtToEquity": info.get("debtToEquity")
        })
    except Exception as e:
        print(f"Could not fetch firm controls for {ticker}: {e}")

firm_controls_df = pd.DataFrame(firm_controls)

print("\nFirm control variables downloaded successfully.")
print(firm_controls_df.head())

# ================================================================
# 9. Merge firm controls with panel data
# ================================================================
merged_final = pd.merge(merged_df, firm_controls_df, on="Ticker", how="left")

# Drop duplicates (if any)
merged_final = merged_final.drop_duplicates(subset=["Ticker", "Year", "Quarter"])

# ================================================================
# 10. Save final enriche dataset
# ================================================================
merged_final.to_csv("covid_dividend_macro_firm_controls.csv", index=False)

print("\n============================================================")
print("FINAL DATASET WITH FIRM CONTROLS SAVED")
print("============================================================")
print(f"Rows: {len(merged_final)} | Columns: {len(merged_final.columns)}")
print("Columns include:\n", list(merged_final.columns))

# ================================================================
# 11. Quick Summary Stats
# ================================================================
print("\nDescriptive statistics of firm control variables:")
print(merged_final[['MarketCap', 'PERatio', 'PB_Ratio', 'Beta', 'DebtToEquity']].describe().round(3))