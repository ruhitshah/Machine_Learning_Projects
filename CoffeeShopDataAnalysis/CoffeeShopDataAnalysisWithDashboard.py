
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load your data (replace 'your_data.csv' with your actual file path)
df = pd.read_csv('shop.csv')

# Data preprocessing
df['transaction_date'] = pd.to_datetime(df['transaction_date'])
df['transaction_time'] = pd.to_datetime(df['transaction_time'], format='%H:%M:%S')
df['transaction_hour'] = df['transaction_time'].dt.hour
df['Month'] = df['transaction_date'].dt.to_period('M')

# Analysis data
monthly_sales = df.groupby('Month')['transaction_qty'].sum()
hourly_sales = df.groupby('transaction_hour')['transaction_qty'].sum()
location_sales = df.groupby('store_location')['transaction_qty'].sum()
store_sales = df.groupby('store_id')['transaction_qty'].sum()
top_products = df.groupby('product_id')['transaction_qty'].sum().nlargest(10)
price_demand = df.groupby('unit_price')['transaction_qty'].sum()
category_preferences = df.groupby('product_category')['transaction_qty'].sum()

# Streamlit app layout
st.title("Sales Data Analysis")

# Monthly Sales
st.subheader("Monthly Sales")
fig, ax = plt.subplots()
monthly_sales.plot(kind='bar', ax=ax)
ax.set_title("Monthly Sales Volume")
ax.set_xlabel("Month")
ax.set_ylabel("Total Quantity Sold")
st.pyplot(fig)

# Hourly Sales
st.subheader("Hourly Sales")
fig, ax = plt.subplots()
hourly_sales.plot(kind='bar', ax=ax)
ax.set_title("Hourly Sales Volume")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Total Quantity Sold")
st.pyplot(fig)

# Location Sales
st.subheader("Sales by Location")
fig, ax = plt.subplots()
location_sales.plot(kind='bar', ax=ax)
ax.set_title("Sales Volume by Store Location")
ax.set_xlabel("Store Location")
ax.set_ylabel("Total Quantity Sold")
st.pyplot(fig)

# Store Sales
st.subheader("Sales by Store")
fig, ax = plt.subplots()
store_sales.plot(kind='bar', ax=ax)
ax.set_title("Sales Volume by Store")
ax.set_xlabel("Store ID")
ax.set_ylabel("Total Quantity Sold")
st.pyplot(fig)

# Top Products
st.subheader("Top 10 Products by Sales Volume")
fig, ax = plt.subplots()
top_products.plot(kind='bar', ax=ax)
ax.set_title("Top 10 Products by Quantity Sold")
ax.set_xlabel("Product ID")
ax.set_ylabel("Total Quantity Sold")
st.pyplot(fig)

# Price Demand
st.subheader("Sales by Unit Price")
fig, ax = plt.subplots()
price_demand.plot(kind='line', ax=ax)
ax.set_title("Demand by Unit Price")
ax.set_xlabel("Unit Price")
ax.set_ylabel("Total Quantity Sold")
st.pyplot(fig)

# Category Preferences
st.subheader("Sales by Product Category")
fig, ax = plt.subplots()
category_preferences.plot(kind='bar', ax=ax)
ax.set_title("Sales Volume by Product Category")
ax.set_xlabel("Product Category")
ax.set_ylabel("Total Quantity Sold")
st.pyplot(fig)

# Run the Streamlit app
# In your terminal, run:
# streamlit run app.py



!pip install streamlit

!pip install pyngrok

# Commented out IPython magic to ensure Python compatibility.
# %%writefile app.py
# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# 
# # Load your data (replace 'shop.csv' with your actual file path)
# df = pd.read_csv('shop.csv')
# 
# # Data preprocessing
# df['transaction_date'] = pd.to_datetime(df['transaction_date'])
# df['transaction_time'] = pd.to_datetime(df['transaction_time'], format='%H:%M:%S')
# df['transaction_hour'] = df['transaction_time'].dt.hour
# df['Month'] = df['transaction_date'].dt.to_period('M')
# 
# # Analysis data
# monthly_sales = df.groupby('Month')['transaction_qty'].sum()
# hourly_sales = df.groupby('transaction_hour')['transaction_qty'].sum()
# location_sales = df.groupby('store_location')['transaction_qty'].sum()
# store_sales = df.groupby('store_id')['transaction_qty'].sum()
# top_products = df.groupby('product_id')['transaction_qty'].sum().nlargest(10)
# price_demand = df.groupby('unit_price')['transaction_qty'].sum()
# category_preferences = df.groupby('product_category')['transaction_qty'].sum()
# 
# # Streamlit app layout
# st.title("Sales Data Analysis")
# 
# # Create a dropdown menu for different visualizations
# option = st.selectbox(
#     'Select the analysis segment to view:',
#     ('Monthly Sales', 'Hourly Sales', 'Sales by Location',
#      'Sales by Store', 'Top 10 Products', 'Demand by Unit Price',
#      'Sales by Product Category')
# )
# 
# # Define a function to plot each selected segment
# def plot_data(title, data, kind='bar', xlabel='', ylabel='Total Quantity Sold'):
#     fig, ax = plt.subplots()
#     data.plot(kind=kind, ax=ax)
#     ax.set_title(title)
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)
#     st.pyplot(fig)
# 
# # Plot based on the selected option
# if option == 'Monthly Sales':
#     plot_data("Monthly Sales Volume", monthly_sales, xlabel="Month")
# elif option == 'Hourly Sales':
#     plot_data("Hourly Sales Volume", hourly_sales, xlabel="Hour of Day")
# elif option == 'Sales by Location':
#     plot_data("Sales Volume by Store Location", location_sales, xlabel="Store Location")
# elif option == 'Sales by Store':
#     plot_data("Sales Volume by Store", store_sales, xlabel="Store ID")
# elif option == 'Top 10 Products':
#     plot_data("Top 10 Products by Quantity Sold", top_products, xlabel="Product ID")
# elif option == 'Demand by Unit Price':
#     plot_data("Demand by Unit Price", price_demand, kind='line', xlabel="Unit Price")
# elif option == 'Sales by Product Category':
#     plot_data("Sales Volume by Product Category", category_preferences, xlabel="Product Category")
#

!pip instasll Streamlit

from pyngrok import ngrok

# Replace 'your_authtoken_here' with your actual ngrok authtoken
!ngrok config add-authtoken 2ofdquAADxEue4fIcfpRsKE13G9_4xGJihYKQ3AQQxCLKsYVa





from pyngrok import ngrok

# Start Streamlit app in the background
!streamlit run app.py &>/dev/null&

# Use the correct configuration format to expose the app
public_url = ngrok.connect("http://localhost:8501")
print(f"Streamlit app is live at {public_url}")


