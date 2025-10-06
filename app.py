import os
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from dotenv import load_dotenv
from strategies import quality_low_volatility

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

def generate_trading_recommendation(df, strategy, symbol="Unknown"):
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
            data = fetch_time_series_data(symbol, frequency)
            
            if data:
                df = process_time_series_data(data, frequency)
                
                if df is not None and not df.empty:
                    st.success(f"✅ Successfully fetched {len(df)} {frequency} data points for {symbol}")
                    
                    # Generate trading recommendation
                    recommendation = generate_trading_recommendation(df, trading_strategy, symbol)
                    
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
            else:
                st.error("Failed to fetch data")

if __name__ == "__main__":
    main()
