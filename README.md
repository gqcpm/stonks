# Alpha Vantage Monthly Data Explorer

A Streamlit app that fetches and visualizes monthly time series data from Alpha Vantage API.

## Features

- 📈 Interactive monthly price charts (candlestick)
- 📊 Volume visualization
- 📋 Data table with latest 12 months
- 📥 CSV download functionality
- 🔍 Support for any stock symbol
- 📱 Responsive design

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Get Alpha Vantage API key:**
   - Visit [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
   - Sign up for a free API key
   - Copy `env.example` to `.env`
   - Add your API key to `.env`:
     ```
     ALPHA_VANTAGE_API_KEY=your_api_key_here
     ```

3. **Run the app:**
   ```bash
   streamlit run app.py
   ```

## Usage

1. Enter a stock symbol (e.g., AAPL, MSFT, GOOGL)
2. Click "Fetch Monthly Data"
3. View interactive charts and data table
4. Download data as CSV if needed

## API Limits

- Free Alpha Vantage API: 5 calls per minute, 500 calls per day
- Paid plans available for higher limits

## Data Available

- Monthly Open, High, Low, Close prices
- Monthly Volume
- Price change calculations
- Interactive candlestick charts
