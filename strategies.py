import pandas as pd
import numpy as np

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
    
    # --- Debug Output ---
    print(f"\n=== Quality/Low Volatility Strategy Debug ===")
    print(f"Ticker: {ticker}")
    print(f"Data points analyzed: {len(df)}")
    print(f"Lookback period: {LOOKBACK_PERIOD}")
    print(f"")
    print(f"VOLATILITY ANALYSIS:")
    print(f"  Stock volatility: {stock_volatility:.4f} ({stock_volatility*100:.2f}%)")
    print(f"  Max allowed: {MAX_VOLATILITY_THRESHOLD:.4f} ({MAX_VOLATILITY_THRESHOLD*100:.2f}%)")
    print(f"  Low volatility check: {low_volatility_check}")
    print(f"")
    print(f"QUALITY ANALYSIS:")
    print(f"  Total return: {total_return:.4f} ({total_return*100:.2f}%)")
    print(f"  Min required: {MIN_RETURN_THRESHOLD:.4f} ({MIN_RETURN_THRESHOLD*100:.2f}%)")
    print(f"  Price stability: {price_stability:.4f} ({price_stability*100:.2f}%)")
    print(f"  Min required: {MIN_PRICE_STABILITY:.4f} ({MIN_PRICE_STABILITY*100:.2f}%)")
    print(f"  High quality check: {high_quality_check}")
    print(f"")
    print(f"MOMENTUM ANALYSIS:")
    if len(df) >= 20:
        print(f"  Recent 10-day avg: {recent_10:.2f}")
        print(f"  Previous 10-day avg: {previous_10:.2f}")
    print(f"  Momentum positive: {momentum_positive}")
    print(f"")
    
    # --- D. Decision Logic ---
    
    if low_volatility_check and high_quality_check and momentum_positive:
        recommendation = "Buy"
    elif not low_volatility_check or not high_quality_check:
        recommendation = "Sell"
    else:
        recommendation = "Hold"
    
    print(f"FINAL DECISION:")
    print(f"  Low volatility: {low_volatility_check}")
    print(f"  High quality: {high_quality_check}")
    print(f"  Positive momentum: {momentum_positive}")
    print(f"  Recommendation: {recommendation}")
    print(f"==========================================\n")
    
    return recommendation