import os
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from dotenv import load_dotenv
from strategies import quality_low_volatility, get_sentiment_scores_by_frequency

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Alpha Vantage Monthly Data",
    page_icon="📈",
    layout="wide"
)

# Alpha Vantage API configuration
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
BASE_URL = "https://www.alphavantage.co/query"

def fetch_time_series_data(symbol, frequency):
    """Fetch time series data from Alpha Vantage based on frequency"""
    if not ALPHA_VANTAGE_API_KEY:
        st.error("❌ Alpha Vantage API key not found. Please set ALPHA_VANTAGE_API_KEY in your .env file")
        return None
    
    # Map frequency to Alpha Vantage function
    function_map = {
        "daily": "TIME_SERIES_DAILY",
        "weekly": "TIME_SERIES_WEEKLY", 
        "monthly": "TIME_SERIES_MONTHLY"
    }
    
    params = {
        "function": function_map[frequency],
        "symbol": symbol,
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    
    try:
        response = requests.get(BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Check for API errors
        if "Error Message" in data:
            st.error(f"❌ API Error: {data['Error Message']}")
            return None
        
        if "Note" in data:
            st.warning(f"⚠️ API Note: {data['Note']}")
            return None
        
        if "Information" in data:
            st.info(f"ℹ️ API Info: {data['Information']}")
            return None
        
        return data
        
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Request failed: {str(e)}")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        return None

def generate_fake_stock_data(symbol, frequency, num_periods=50):
    """Generate fake stock data for testing purposes"""
    import random
    import numpy as np
    
    # Set seed for reproducible data
    random.seed(hash(symbol) % 1000)
    
    # Generate date range based on frequency
    end_date = datetime.now()
    if frequency == "daily":
        start_date = end_date - timedelta(days=num_periods)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    elif frequency == "weekly":
        start_date = end_date - timedelta(weeks=num_periods)
        date_range = pd.date_range(start=start_date, end=end_date, freq='W')
    else:  # monthly
        start_date = end_date - timedelta(days=num_periods * 30)
        date_range = pd.date_range(start=start_date, end=end_date, freq='M')
    
    # Generate realistic stock prices
    base_price = {"AAPL": 150, "MSFT": 300, "GOOGL": 2500, "TSLA": 200}.get(symbol, 100)
    prices = []
    current_price = base_price
    
    for i in range(len(date_range)):
        # Add some trend and volatility
        change = random.gauss(0, 0.02)  # 2% daily volatility
        current_price *= (1 + change)
        prices.append(current_price)
    
    # Create OHLCV data
    data = []
    for i, (date, close) in enumerate(zip(date_range, prices)):
        # Generate realistic OHLC from close price
        volatility = random.uniform(0.01, 0.03)
        high = close * (1 + volatility)
        low = close * (1 - volatility)
        open_price = close * random.uniform(0.99, 1.01)
        volume = random.randint(1000000, 10000000)
        
        data.append({
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume
        })
    
    df = pd.DataFrame(data, index=date_range)
    return df

def process_time_series_data(data, frequency):
    """Process the time series data into a pandas DataFrame"""
    # Map frequency to data key
    data_key_map = {
        "daily": "Time Series (Daily)",
        "weekly": "Weekly Time Series",
        "monthly": "Monthly Time Series"
    }
    
    data_key = data_key_map[frequency]
    
    if not data or data_key not in data:
        return None
    
    time_series = data[data_key]
    
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(time_series, orient='index')
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    
    # Rename columns for better readability
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Convert to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def calculate_moving_average(df, window):
    """Calculate moving average for the given window"""
    return df['Close'].rolling(window=window).mean()

def generate_trading_recommendation(df, strategy, symbol="Unknown", frequency="monthly"):
    """Generate trading recommendation based on selected strategy"""
    if strategy == "Quality/Low Volatility":
        try:
            # Use the implemented quality_low_volatility strategy
            return quality_low_volatility(df, symbol)
        except Exception as e:
            st.warning(f"Strategy calculation error: {str(e)}")
            return "Hold"
    
    return "Hold"  # Default recommendation

def create_price_chart(df, symbol, frequency, indicator="None"):
    """Create an interactive price chart with optional indicators"""
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name=f"{symbol} {frequency.title()}"
    ))
    
    # Add technical indicators
    if indicator == "50 Day Moving Average":
        # Calculate 50-day moving average
        ma_50 = calculate_moving_average(df, 50)
        fig.add_trace(go.Scatter(
            x=df.index,
            y=ma_50,
            mode='lines',
            name='50 Day MA',
            line=dict(color='orange', width=2)
        ))
    
    fig.update_layout(
        title=f"{symbol} {frequency.title()} Price Data",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=600,
        showlegend=True
    )
    
    return fig

def create_volume_chart(df, symbol, frequency):
    """Create a volume chart"""
    fig = px.bar(
        x=df.index,
        y=df['Volume'],
        title=f"{symbol} {frequency.title()} Volume",
        labels={'x': 'Date', 'y': 'Volume'}
    )
    
    fig.update_layout(height=400)
    return fig

def create_sentiment_grid(sentiment_scores, symbol, frequency):
    """Create a GitHub-style contributions grid for sentiment data"""
    import numpy as np
    
    # Convert sentiment scores to a grid format
    dates = sentiment_scores.index
    scores = sentiment_scores.values
    
    # Create a grid layout (similar to GitHub contributions)
    # Group by weeks for better visualization
    if frequency == "daily":
        # For daily data, create a weekly grid
        weeks = []
        current_week = []
        
        for i, (date, score) in enumerate(zip(dates, scores)):
            current_week.append((date, score))
            
            # Start new week on Monday or when we have 7 days
            if date.weekday() == 6 or len(current_week) == 7:  # Sunday or 7 days
                weeks.append(current_week)
                current_week = []
        
        # Add remaining days
        if current_week:
            weeks.append(current_week)
            
    elif frequency == "weekly":
        # For weekly data, each row is a month
        weeks = []
        current_month = []
        current_month_num = None
        
        for date, score in zip(dates, scores):
            month_num = date.month
            
            if current_month_num is None:
                current_month_num = month_num
            elif month_num != current_month_num:
                weeks.append(current_month)
                current_month = []
                current_month_num = month_num
            
            current_month.append((date, score))
        
        if current_month:
            weeks.append(current_month)
            
    else:  # monthly
        # For monthly data, each row is a year
        weeks = []
        current_year = []
        current_year_num = None
        
        for date, score in zip(dates, scores):
            year_num = date.year
            
            if current_year_num is None:
                current_year_num = year_num
            elif year_num != current_year_num:
                weeks.append(current_year)
                current_year = []
                current_year_num = year_num
            
            current_year.append((date, score))
        
        if current_year:
            weeks.append(current_year)
    
    # Create the grid visualization
    fig = go.Figure()
    
    # Color mapping function (GitHub-style)
    def get_sentiment_color(score):
        """Convert sentiment score to GitHub-style colors"""
        if score > 0:
            # Positive: green scale (GitHub contribution style)
            intensity = min(abs(score), 1.0)
            if intensity > 0.8:
                return "#0e4429"  # Darkest green
            elif intensity > 0.6:
                return "#006d32"  # Dark green
            elif intensity > 0.4:
                return "#26a641"  # Medium green
            elif intensity > 0.2:
                return "#39d353"  # Light green
            else:
                return "#9be9a8"  # Lightest green
        elif score < 0:
            # Negative: red scale
            intensity = min(abs(score), 1.0)
            if intensity > 0.8:
                return "#8b0000"  # Darkest red
            elif intensity > 0.6:
                return "#dc2626"  # Dark red
            elif intensity > 0.4:
                return "#ef4444"  # Medium red
            elif intensity > 0.2:
                return "#f87171"  # Light red
            else:
                return "#fca5a5"  # Lightest red
        else:
            # Neutral: light gray (like GitHub's empty days)
            return "#ebedf0"
    
    # Add squares for each sentiment score
    y_pos = 0
    for week in weeks:
        x_pos = 0
        for date, score in week:
            color = get_sentiment_color(score)
            
            fig.add_trace(go.Scatter(
                x=[x_pos],
                y=[y_pos],
                mode='markers',
                marker=dict(
                    size=11,  # GitHub's exact square size
                    color=color,
                    line=dict(width=0, color='rgba(0,0,0,0)'),
                    symbol='square'
                ),
                hovertemplate=f"Date: {date.strftime('%Y-%m-%d')}<br>Sentiment: {score:.3f}<extra></extra>",
                showlegend=False
            ))
            x_pos += 1
        y_pos += 1
    
    # Update layout (GitHub-style)
    fig.update_layout(
        title=dict(
            text=f"{symbol} {frequency.title()} Sentiment Grid",
            font=dict(size=16, color='#24292f'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            fixedrange=True,
            range=[-0.5, max(6, len(weeks[0]) if weeks else 6) - 0.5]
        ),
        yaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            scaleanchor="x",
            scaleratio=1,
            fixedrange=True,
            range=[-0.5, len(weeks) - 0.5]
        ),
        height=max(150, len(weeks) * 15),  # GitHub's compact height
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background to match Streamlit
        margin=dict(l=5, r=5, t=40, b=5),
        font=dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif"),
        showlegend=False
    )
    
    return fig

# Main Streamlit app
def main():
    st.title("📈 Alpha Vantage Time Series Data Explorer")
    st.markdown("Get and visualize daily, weekly, or monthly time series data for any stock symbol")
    
    # Sidebar for input
    with st.sidebar:
        st.header("Settings")
        symbol = st.text_input(
            "Stock Symbol",
            value="AAPL",
            help="Enter a stock symbol (e.g., AAPL, MSFT, GOOGL)"
        ).upper()
        
        frequency = st.selectbox(
            "Time Series Frequency",
            options=["daily", "weekly", "monthly"],
            index=2,  # Default to monthly
            help="Choose the frequency of data points"
        )
        
        indicator = st.selectbox(
            "Technical Indicator",
            options=["None", "50 Day Moving Average"],
            index=0,  # Default to None
            help="Choose a technical indicator to overlay on the price chart"
        )
        
        trading_strategy = st.selectbox(
            "Trading Strategy",
            options=["Quality/Low Volatility"],
            index=0,  # Default to Quality/Low Volatility
            help="Choose a trading strategy for generating buy/sell/hold recommendations"
        )
        
        include_sentiment = st.checkbox(
            "Include Sentiment Analysis",
            value=False,
            help="Analyze news sentiment for the selected time period (requires additional API calls)"
        )
        
        use_fake_sentiment = st.checkbox(
            "Use Fake Sentiment Data",
            value=True,
            help="Generate realistic fake sentiment data for testing (avoids API limits)"
        )
        
        use_fake_stock_data = st.checkbox(
            "Use Fake Stock Data",
            value=True,
            help="Generate realistic fake stock data for testing (bypasses API limits)"
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("This app fetches time series data from Alpha Vantage API and displays it with interactive charts.")
        
        if not ALPHA_VANTAGE_API_KEY:
            st.error("⚠️ API Key Required")
            st.markdown("Get your free API key from [Alpha Vantage](https://www.alphavantage.co/support/#api-key)")
    
    # Main content
    if st.button(f"📊 Fetch {frequency.title()} Data", type="primary"):
        if not symbol:
            st.error("Please enter a stock symbol")
            return
        
        with st.spinner(f"Fetching {frequency} data for {symbol}..."):
            if use_fake_stock_data:
                # Generate fake stock data
                st.info("🎭 Using fake stock data for testing...")
                df = generate_fake_stock_data(symbol, frequency)
            else:
                # Fetch real data from API
                data = fetch_time_series_data(symbol, frequency)
                
                if data:
                    df = process_time_series_data(data, frequency)
                else:
                    df = None
            
            if df is not None and not df.empty:
                st.success(f"✅ Successfully fetched {len(df)} {frequency} data points for {symbol}")
                
                # Generate trading recommendation
                recommendation = generate_trading_recommendation(df, trading_strategy, symbol, frequency)
                
                # Display basic statistics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    price_change = df['Close'].iloc[-1] - df['Close'].iloc[-2] if len(df) > 1 else 0
                    st.metric("Latest Close", f"${df['Close'].iloc[-1]:.2f}", f"{price_change:.2f}")
                
                with col2:
                    st.metric("Highest Price", f"${df['High'].max():.2f}")
                
                with col3:
                    st.metric("Lowest Price", f"${df['Low'].min():.2f}")
                
                with col4:
                    # Color code the recommendation
                    if recommendation == "Buy":
                        st.metric("Recommendation", recommendation, delta_color="normal")
                    elif recommendation == "Sell":
                        st.metric("Recommendation", recommendation, delta_color="inverse")
                    else:
                        st.metric("Recommendation", recommendation, delta_color="off")
                
                st.markdown("---")
                
                # Create charts
                st.subheader("📊 Price Chart")
                price_chart = create_price_chart(df, symbol, frequency, indicator)
                st.plotly_chart(price_chart, use_container_width=True)
                
                st.subheader("📊 Volume Chart")
                volume_chart = create_volume_chart(df, symbol, frequency)
                st.plotly_chart(volume_chart, use_container_width=True)
                
                st.markdown("---")
                
                # Display data table with appropriate number of rows
                display_rows = 30 if frequency == "daily" else (12 if frequency == "weekly" else 12)
                st.subheader(f"📋 {frequency.title()} Data Table")
                st.dataframe(df.tail(display_rows), use_container_width=True)
                
                # Sentiment Analysis Section
                if include_sentiment:
                    st.markdown("---")
                    st.subheader("📰 Sentiment Analysis")
                    
                    with st.spinner(f"Analyzing {frequency} sentiment for {symbol}..."):
                        try:
                            sentiment_scores = get_sentiment_scores_by_frequency(
                                symbol, df, frequency, ALPHA_VANTAGE_API_KEY, use_fake_sentiment
                            )
                            
                            if not sentiment_scores.empty:
                                # Display sentiment metrics
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    avg_sentiment = sentiment_scores.mean()
                                    st.metric("Average Sentiment", f"{avg_sentiment:.3f}")
                                
                                with col2:
                                    recent_sentiment = sentiment_scores.tail(5).mean()
                                    st.metric("Recent Sentiment", f"{recent_sentiment:.3f}")
                                
                                with col3:
                                    positive_days = (sentiment_scores > 0.1).sum()
                                    total_days = len(sentiment_scores)
                                    st.metric("Positive Days", f"{positive_days}/{total_days}")
                                
                                # Create sentiment grid chart
                                st.subheader("📊 Sentiment Grid")
                                sentiment_grid = create_sentiment_grid(sentiment_scores, symbol, frequency)
                                st.plotly_chart(sentiment_grid, use_container_width=True)
                                
                                # Display sentiment data table
                                st.subheader("📋 Sentiment Data")
                                st.dataframe(sentiment_scores.to_frame(), use_container_width=True)
                                
                            else:
                                st.warning("No sentiment data available for this period")
                                
                        except Exception as e:
                            st.error(f"Sentiment analysis failed: {str(e)}")
                
                # Download option
                csv = df.to_csv()
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"{symbol}_{frequency}_data.csv",
                    mime="text/csv"
                )
                
            else:
                st.error("No data available for this symbol")

if __name__ == "__main__":
    main()
