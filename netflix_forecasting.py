import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import numpy as np
import datetime

# Load the cleaned data
df = pd.read_csv("netflix_cleaned.csv")

# Convert date_added to datetime if not already
df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
df = df.dropna(subset=['date_added'])

# Group by year
df['year_added'] = df['date_added'].dt.year
yearly_trend = df.groupby('year_added').size().reset_index(name='content_count')

plt.figure(figsize=(10, 5))
sns.lineplot(data=yearly_trend, x='year_added', y='content_count', marker='o')
plt.title("Netflix Content Added Over the Years")
plt.xlabel("Year")
plt.ylabel("Number of Titles")
plt.grid(True)
plt.show()

# Prepare data for ML model
X = yearly_trend['year_added'].values.reshape(-1, 1)
y = yearly_trend['content_count'].values

# Train Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Predict for next 5 years
future_years = np.arange(2025, 2031).reshape(-1, 1)
future_preds = model.predict(future_years)

# Combine predictions with real data
future_df = pd.DataFrame({'year_added': future_years.flatten(), 'content_count': future_preds.astype(int)})
combined = pd.concat([yearly_trend, future_df])

plt.figure(figsize=(10, 5))
sns.lineplot(data=combined, x='year_added', y='content_count', label='Forecast', marker='o')
plt.axvline(x=2024.5, linestyle='--', color='gray', label='Prediction Starts')
plt.title("Forecast of Netflix Content Additions")
plt.xlabel("Year")
plt.ylabel("Titles Added")
plt.legend()
plt.grid(True)
plt.show()

latest_year = int(yearly_trend['year_added'].max())
latest_content = int(yearly_trend[yearly_trend['year_added'] == latest_year]['content_count'])

growth_rate = (model.coef_[0] / latest_content) * 100

summary = f"""
📈 Based on past data, Netflix has been adding more content every year, peaking at {latest_content} titles in {latest_year}.
🧠 Our ML model predicts Netflix could add over {int(future_preds[-1])} titles by 2030 if the trend continues.
📊 Estimated annual growth rate in content is around {growth_rate:.2f}%.
🎯 Recommendation: Netflix is diversifying rapidly — genre segmentation and country-specific content are potential growth hotspots.
"""

print(summary)
