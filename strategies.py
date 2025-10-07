import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime, timedelta
import time

def get_sentiment_score_for_date(ticker, date, api_key=None):
    """
    Get sentiment score for a specific date using Alpha Vantage News Sentiment API.
    
    Args:
        ticker (str): Stock ticker symbol
        date (str): Date in YYYY-MM-DD format
        api_key (str): Alpha Vantage API key (optional, will use environment variable if not provided)
    
    Returns:
        float: Average sentiment score (-1 to 1, where 1 is most positive)
    """
    if not api_key:
        api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    
    if not api_key:
        print(f"Warning: No ALPHA_VANTAGE_API_KEY found for {ticker} on {date}")
        return 0.0  # Neutral sentiment if no API key
    
    # Alpha Vantage News Sentiment endpoint
    url = "https://www.alphavantage.co/query"
    
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "time_from": f"{date}T00:00:00",
        "time_to": f"{date}T23:59:59",
        "limit": 5,  # Top 5 most relevant articles
        "apikey": api_key
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Check for API errors
        if "Error Message" in data:
            print(f"Alpha Vantage Error: {data['Error Message']}")
            return 0.0
        
        if "Note" in data:
            print(f"Alpha Vantage Note: {data['Note']}")
            return 0.0
        
        if "Information" in data:
            print(f"Alpha Vantage Info: {data['Information']}")
            return 0.0
        
        if "feed" not in data or not data["feed"]:
            return 0.0  # No news articles found
        
        articles = data["feed"]
        sentiment_scores = []
        
        for article in articles:
            # Alpha Vantage provides pre-calculated sentiment scores
            if "overall_sentiment_score" in article:
                sentiment = float(article["overall_sentiment_score"])
                sentiment_scores.append(sentiment)
        
        if sentiment_scores:
            return np.mean(sentiment_scores)
        else:
            return 0.0  # No valid sentiment scores found
            
    except Exception as e:
        print(f"Error fetching news sentiment for {ticker} on {date}: {str(e)}")
        return 0.0

def _get_date_for_frequency(date, frequency):
    """
    Get the appropriate date for sentiment analysis based on frequency.
    
    Args:
        date: Date object
        frequency (str): Time series frequency ("daily", "weekly", "monthly")
    
    Returns:
        str: Formatted date string for API call
    """
    if frequency == "daily":
        return date.strftime("%Y-%m-%d")
    elif frequency == "weekly":
        # Get Monday of the week for this date
        monday = date - timedelta(days=date.weekday())
        return monday.strftime("%Y-%m-%d")
    elif frequency == "monthly":
        # Get first day of the month for this date
        first_of_month = date.replace(day=1)
        return first_of_month.strftime("%Y-%m-%d")
    else:
        raise ValueError(f"Unsupported frequency: {frequency}")

def _get_progress_interval(frequency):
    """Get progress reporting interval based on frequency."""
    return {"daily": 10, "weekly": 5, "monthly": 3}.get(frequency, 10)

def get_sentiment_scores_by_frequency(ticker, df, frequency, api_key=None):
    """
    Get sentiment scores based on the time series frequency (daily, weekly, monthly).
    
    Args:
        ticker (str): Stock ticker symbol
        df (pd.DataFrame): DataFrame with datetime index
        frequency (str): Time series frequency ("daily", "weekly", "monthly")
        api_key (str): Alpha Vantage API key (optional)
    
    Returns:
        pd.Series: Sentiment scores indexed by date, aggregated by frequency
    """
    # Validate frequency
    valid_frequencies = {"daily", "weekly", "monthly"}
    if frequency not in valid_frequencies:
        raise ValueError(f"Invalid frequency: {frequency}. Must be one of {valid_frequencies}")
    
    # Get unique dates from dataframe
    unique_dates = df.index.date
    total_dates = len(unique_dates)
    
    print(f"Fetching {frequency} sentiment data for {ticker} across {total_dates} periods...")
    
    sentiment_scores = []
    dates = []
    progress_interval = _get_progress_interval(frequency)
    
    # Process each date
    for i, date in enumerate(unique_dates):
        # Get appropriate date string for this frequency
        date_str = _get_date_for_frequency(date, frequency)
        
        # Add small delay to respect API rate limits (except for first call)
        if i > 0:
            time.sleep(0.1)  # 100ms delay between requests
        
        # Fetch sentiment for this date
        sentiment = get_sentiment_score_for_date(ticker, date_str, api_key)
        sentiment_scores.append(sentiment)
        dates.append(date)
        
        # Progress indicator
        if (i + 1) % progress_interval == 0:
            print(f"Processed {i + 1}/{total_dates} {frequency} periods...")
    
    # Create Series with datetime index
    sentiment_series = pd.Series(sentiment_scores, index=pd.to_datetime(dates))
    sentiment_series.name = f"{ticker}_{frequency}_sentiment"
    
    print(f"Completed {frequency} sentiment analysis for {ticker}")
    return sentiment_series

def quality_low_volatility(df, ticker="Unknown"):
    """
    Quality/Low Volatility Strategy
    
    This strategy identifies stocks that are of high quality and have low volatility.
    It uses price data to calculate volatility metrics and makes recommendations based on
    risk-adjusted returns and price stability.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data and datetime index
        
    Returns:
        str: "Buy", "Sell", or "Hold" recommendation
    """
    
    # --- 1. Define Thresholds & Lookback Periods ---
    LOOKBACK_PERIOD = min(252, len(df))  # 252 trading days for one year, or available data
    MIN_RETURN_THRESHOLD = 0.02  # Minimum 2% return over lookback period
    MAX_VOLATILITY_THRESHOLD = 1.6  # Maximum 160% annualized volatility
    MIN_PRICE_STABILITY = 0.003  # Minimum 0.3% of days should be within 3% of average price
    
    if len(df) < 30:  # Need at least 30 days of data
        return "Hold"
    
    # --- A. Calculate Volatility (Risk) ---
    
    # 1. Calculate daily returns
    df['Returns'] = df['Close'].pct_change().dropna()
    
    # 2. Calculate annualized volatility (standard deviation of returns * sqrt(252))
    stock_volatility = df['Returns'].std() * np.sqrt(252)
    
    # 3. Check if volatility is within acceptable limits
    low_volatility_check = stock_volatility < MAX_VOLATILITY_THRESHOLD
    
    # --- B. Calculate Quality (Price Stability & Returns) ---
    
    # 1. Calculate total return over lookback period
    lookback_data = df.tail(LOOKBACK_PERIOD)
    if len(lookback_data) < 2:
        return "Hold"
    
    total_return = (lookback_data['Close'].iloc[-1] - lookback_data['Close'].iloc[0]) / lookback_data['Close'].iloc[0]
    
    # 2. Calculate price stability (percentage of days within 3% of average price)
    avg_price = lookback_data['Close'].mean()
    price_stability = (abs(lookback_data['Close'] - avg_price) / avg_price < 0.03).mean()
    
    # 3. Check if quality metrics are within acceptable limits
    high_quality_check = (total_return >= MIN_RETURN_THRESHOLD) and (price_stability >= MIN_PRICE_STABILITY)
    
    # --- C. Additional Momentum Check ---
    
    # Check recent momentum (last 10 days vs previous 10 days)
    if len(df) >= 20:
        recent_10 = df['Close'].tail(10).mean()
        previous_10 = df['Close'].tail(20).head(10).mean()
        momentum_positive = recent_10 > previous_10
    else:
        momentum_positive = True  # Default to positive if not enough data
    
    # --- D. Decision Logic ---
    
    if low_volatility_check and high_quality_check and momentum_positive:
        recommendation = "Buy"
    elif not low_volatility_check or not high_quality_check:
        recommendation = "Sell"
    else:
        recommendation = "Hold"
    
    return recommendation