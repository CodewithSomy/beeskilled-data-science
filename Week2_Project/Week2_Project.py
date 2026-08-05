import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# ==========================================
# 1. LOAD DATA
# ==========================================

folder_path = r"Week2_Project"

full_grouped = pd.read_csv(
    folder_path + r"\full_grouped.csv"
)

country_latest = pd.read_csv(
    folder_path + r"\country_wise_latest.csv"
)

# ==========================================
# 2. PREPARE DATA
# ==========================================

full_grouped['Date'] = pd.to_datetime(
    full_grouped['Date']
)

million_formatter = FuncFormatter(
    lambda x, pos: f'{x / 1_000_000:.1f}M'
)


# ==========================================
# 3. FIND TOP 5 COUNTRIES
# ==========================================

latest_date = full_grouped['Date'].max()

latest_data = full_grouped[
    full_grouped['Date'] == latest_date
]

top5 = latest_data.nlargest(5, 'Confirmed')

# ==========================================
# 4. COUNTRIES FOR TREND ANALYSIS
# ==========================================

top5_countries = top5['Country/Region'].tolist()

# ==========================================
# 5. TREND LINE - CONFIRMED CASES OVER TIME
# ==========================================

plt.figure(figsize=(12, 6))

for country in top5_countries:

    country_data = full_grouped[
        full_grouped['Country/Region'] == country
    ]

    plt.plot(
        country_data['Date'],
        country_data['Confirmed'],
        label=country
    )

plt.gca().yaxis.set_major_formatter(million_formatter)

plt.title("COVID-19 Confirmed Cases Over Time")
plt.xlabel("Date")
plt.ylabel("Confirmed Cases (Millions)")
plt.legend()
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# ==========================================
# 6. TOP 5 COUNTRIES BAR GRAPH
# ==========================================
plt.figure(figsize=(10, 6))

plt.bar(
    top5['Country/Region'],
    top5['Confirmed']
)

plt.gca().yaxis.set_major_formatter(million_formatter)

plt.title("Top 5 Countries by Confirmed COVID-19 Cases")
plt.xlabel("Country")
plt.ylabel("Confirmed Cases (Millions)")

plt.xticks(rotation=30)

plt.tight_layout()
plt.show()

# ==========================================
# 7. HEATMAP - TOP 5 COUNTRIES
# ==========================================

heatmap_data = full_grouped[
    full_grouped['Country/Region'].isin(top5_countries)
].pivot(
    index='Country/Region',
    columns='Date',
    values='Confirmed'
)

plt.figure(figsize=(16, 6))

sns.heatmap(
    heatmap_data,
    cmap='turbo'
)

plt.title("COVID-19 Confirmed Cases - Top 5 Countries")
plt.xlabel("Date")
plt.ylabel("Country")

# Clean up axis labels
plt.yticks(rotation=0)

plt.xticks(
    ticks=range(0, len(heatmap_data.columns), 20),
    labels=[
        heatmap_data.columns[i].strftime('%d-%b')
        for i in range(0, len(heatmap_data.columns), 20)
    ],
    rotation=45
)

plt.tight_layout()
plt.show()

# ==========================================
# 8. SCATTER PLOT - CASES VS DEATHS
# ==========================================

plt.figure(figsize=(10, 6))

plt.scatter(
    country_latest['Confirmed'],
    country_latest['Deaths'],
    alpha=0.6
)

plt.xscale('log')
plt.yscale('log')


plt.title("Confirmed COVID-19 Cases vs Deaths")
plt.xlabel("Confirmed Cases")
plt.ylabel("Deaths")

plt.tight_layout()
plt.show()

# ==========================================
# 9. KEY OBSERVATIONS
# ==========================================

print("\nKey Observations:")
print(f"Analysis period: {full_grouped['Date'].min().date()} to {latest_date.date()}")
print("The US had the highest confirmed cases on the final date.")
print("The Top 5 countries showed substantial differences in COVID-19 case growth.")
print("The scatter plot indicates a positive relationship between confirmed cases and deaths.")
print("The heatmap shows the increasing intensity of confirmed cases over time.")