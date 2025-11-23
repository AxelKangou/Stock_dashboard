import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import date, timedelta
import plotly.graph_objects as go
from scipy.signal import argrelextrema # For finding local extrema

# --- Configuration ---
st.set_page_config(
    layout="wide", 
    page_title="Stock Performance Dashboard with S&R", 
    menu_items=None
)

# --- Constants ---
DEFAULT_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 
    'NVDA', 'JPM', 'V', 'SPY', 'QQQ'
]
MAX_SELECTIONS = 9
CHART_COLUMNS = 3

# --- Data Fetching Function (Cached) ---
@st.cache_data(ttl=3600)
def get_stock_data(tickers, start_date, end_date):
    """Fetches OHLCV data for a list of tickers from Yahoo Finance."""
    if not tickers:
        return pd.DataFrame()
    data = yf.download(tickers, start=start_date, end=end_date)
    return data.dropna()

# --- NEW: Function to find simplified Support and Resistance ---
def find_support_resistance(df, window=20, num_levels=3):
    """
    Identifies simplified support and resistance levels using local extrema.
    Higher window means less sensitive. Num_levels limits the plotted S&R.
    """
    if df.empty or 'Close' not in df.columns:
        return [], []

    # Convert Series to numpy array for scipy.signal
    prices = df['Close'].values

    # Find local maxima (potential resistance)
    resistance_indices = argrelextrema(prices, comparator=lambda x, y: x > y, order=window)[0]
    resistance_levels = prices[resistance_indices]
    
    # Find local minima (potential support)
    support_indices = argrelextrema(prices, comparator=lambda x, y: x < y, order=window)[0]
    support_levels = prices[support_indices]

    # Sort and take unique levels (optional: filter for distinct values)
    # For a simplified approach, we can just sort and pick top/bottom `num_levels`
    resistance_levels = sorted(list(set(resistance_levels)), reverse=True)
    support_levels = sorted(list(set(support_levels)))

    # Limit the number of levels to plot
    resistance_levels = resistance_levels[:num_levels]
    support_levels = support_levels[:num_levels]

    return support_levels, resistance_levels

# --- Plotting Function (Modified to include S&R) ---
def plot_candlestick(df, ticker, height=300, sr_window=20, sr_num_levels=3, show_sma=False, sma_period=20):
    """Generates a Plotly Candlestick chart for a single ticker with S&R."""
    try:
        if isinstance(df.columns, pd.MultiIndex):
            plot_data = df.loc[:, (slice(None), ticker)]
            plot_data.columns = plot_data.columns.droplevel(1)
        else:
            plot_data = df
            
        fig = go.Figure(data=[go.Candlestick(
            x=plot_data.index,
            open=plot_data['Open'],
            high=plot_data['High'],
            low=plot_data['Low'],
            close=plot_data['Close'],
            name='Candlestick'
        )])

        # Add Moving Average if selected
        if show_sma and 'Close' in plot_data.columns:
            plot_data[f'SMA_{sma_period}'] = plot_data['Close'].rolling(window=sma_period).mean()
            fig.add_trace(go.Scatter(
                x=plot_data.index,
                y=plot_data[f'SMA_{sma_period}'],
                mode='lines',
                name=f'SMA {sma_period}',
                line=dict(color='blue', width=1)
            ))

        # Add Support and Resistance levels
        support_levels, resistance_levels = find_support_resistance(plot_data, window=sr_window, num_levels=sr_num_levels)

        for s_level in support_levels:
            fig.add_hline(y=s_level, annotation_text=f"S: {s_level:.2f}", 
                          annotation_position="bottom right", line_dash="dot", 
                          line_color="green", line_width=1)
        for r_level in resistance_levels:
            fig.add_hline(y=r_level, annotation_text=f"R: {r_level:.2f}", 
                          annotation_position="top right", line_dash="dot", 
                          line_color="red", line_width=1)

        fig.update_layout(
            title=f'{ticker} Candlestick Chart',
            xaxis_title='Date',
            yaxis_title='Price',
            xaxis_rangeslider_visible=False,
            height=height,
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(x=0, y=0.99, bgcolor='rgba(255,255,255,0)', bordercolor='rgba(255,255,255,0)')
        )
        st.plotly_chart(fig, use_container_width=True)

    except KeyError:
        st.warning(f"Required OHLC data is missing for **{ticker}** in this range.")


# --- Application Logic ---
st.title('📈 Stock Performance Grid Analyzer (Candlestick with S&R)')
st.markdown('Select up to 9 assets to view their performance in a dynamic candlestick grid with calculated Support and Resistance levels.')

# --- User Inputs (Sidebar) ---
with st.sidebar:
    st.header("Configuration")
    
    selected_tickers = st.multiselect(
        'Select Assets', 
        DEFAULT_TICKERS,
        default=DEFAULT_TICKERS[:3],
        max_selections=MAX_SELECTIONS,
        placeholder="Choose up to 9 tickers"
    )
    
    today = date.today()
    default_start = today - timedelta(days=365)
    
    start_date = st.date_input(
        'Start Date',
        value=default_start,
        max_value=today,
        help="Data is fetched from this date."
    )
    
    end_date = st.date_input(
        'End Date', 
        value=today,
        max_value=today,
        help="Data is fetched up to this date."
    )

    st.subheader("Technical Indicators")
    show_sma = st.checkbox("Show Simple Moving Average (SMA)", value=False)
    sma_period = st.slider("SMA Period", min_value=10, max_value=200, value=20, step=5, disabled=not show_sma)

    st.subheader("Support & Resistance Settings")
    sr_enabled = st.checkbox("Show Support & Resistance", value=True)
    if sr_enabled:
        sr_window = st.slider("S&R Detection Window (Sensitivity)", min_value=5, max_value=50, value=20, step=1,
                              help="Smaller window means more S&R levels (more sensitive).")
        sr_num_levels = st.slider("Number of S&R Levels to Display", min_value=1, max_value=10, value=3, step=1,
                                  help="Controls how many significant S&R lines are drawn.")
    else:
        sr_window = 20 # Default if disabled
        sr_num_levels = 0 # No levels if disabled


# --- Main Content ---
if not selected_tickers:
    st.info("👈 Please select at least one asset in the sidebar to begin plotting.")
elif start_date >= end_date:
    st.error("The Start Date must be before the End Date. Please adjust the dates in the sidebar.")
else:
    try:
        data = get_stock_data(selected_tickers, start_date, end_date)
    except Exception as e:
        st.error(f"Error fetching data: {e}. Check your selected dates and tickers.")
        data = pd.DataFrame()

    if data.empty and selected_tickers:
        st.warning("No data returned for the selected tickers/date range. Data might be missing or the market is closed.")
    elif not data.empty:
        st.markdown(f"**Viewing data from {start_date} to {end_date}**")
        
        cols = st.columns(CHART_COLUMNS)
        
        for i, ticker in enumerate(selected_tickers):
            current_col = cols[i % CHART_COLUMNS]
            
            with current_col:
                st.subheader(f"📊 {ticker}")
                
                if (('Close', ticker) in data.columns) or (ticker in data.columns):
                    # Pass S&R parameters from sidebar
                    plot_candlestick(data, ticker, height=300, 
                                     sr_window=sr_window if sr_enabled else 0, # Pass 0 if disabled
                                     sr_num_levels=sr_num_levels if sr_enabled else 0, # Pass 0 if disabled
                                     show_sma=show_sma,
                                     sma_period=sma_period)
                else:
                    st.warning(f"OHLC data not available for **{ticker}** in this range.")