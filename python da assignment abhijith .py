"""
Python DA Assignment 2 - Data Visualization
Dataset: seaborn 'taxis'
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# ---------------------------------------------------------------
# 1) Loading the Taxis Dataset
# ---------------------------------------------------------------
df = sns.load_dataset("taxis")
print("Shape:", df.shape)
print(df.head())
print(df.info())

# ---------------------------------------------------------------
# 2) Handling Missing Values
# ---------------------------------------------------------------
# a) Check for missing values
missing = df.isnull().sum()
print("\nMissing values per column:")
print(missing[missing > 0])

# b) Drop rows where CRITICAL columns are missing (cannot be reasonably imputed)
critical_cols = ["pickup", "dropoff", "fare", "total"]
df = df.dropna(subset=critical_cols)

# c) Impute remaining columns
#    - numerical columns -> median
#    - categorical columns -> mode
num_cols = df.select_dtypes(include="number").columns
cat_cols = df.select_dtypes(include=["object", "category"]).columns.difference(
    ["pickup", "dropoff"]
)

for col in num_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].median())

for col in cat_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing values after cleaning:", df.isnull().sum().sum())

# Convert pickup / dropoff to datetime
df["pickup"] = pd.to_datetime(df["pickup"])
df["dropoff"] = pd.to_datetime(df["dropoff"])

# ---------------------------------------------------------------
# 3) Visualizations using Matplotlib / Pandas Plot
# ---------------------------------------------------------------

# --- Line Chart: fare over time ---
line_df = df.sort_values("pickup")
plt.figure(figsize=(12, 5))
plt.plot(line_df["pickup"], line_df["fare"], color="steelblue", linewidth=0.8)
plt.title("Fare Over Time")
plt.xlabel("Pickup Timestamp")
plt.ylabel("Fare ($)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("01_line_chart.png", dpi=150)
plt.show()

# --- Bar Chart: total fare per pickup_borough ---
borough_fare = df.groupby("pickup_borough")["fare"].sum().sort_values(ascending=False)
plt.figure(figsize=(8, 5))
borough_fare.plot(kind="bar", color=["#4c72b0", "#55a868", "#c44e52", "#8172b2"])
plt.title("Total Fare by Pickup Borough")
plt.xlabel("Pickup Borough")
plt.ylabel("Total Fare ($)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("02_bar_chart.png", dpi=150)
plt.show()

# --- Pie Chart: trips by payment method ---
payment_counts = df["payment"].value_counts()
plt.figure(figsize=(6, 6))
payment_counts.plot(
    kind="pie", autopct="%1.1f%%", startangle=90,
    colors=["#4c72b0", "#dd8452"], ylabel=""
)
plt.title("Distribution of Trips by Payment Method")
plt.tight_layout()
plt.savefig("03_pie_chart.png", dpi=150)
plt.show()

# --- Histogram: distribution of distance ---
plt.figure(figsize=(9, 5))
plt.hist(df["distance"], bins=40, color="teal", edgecolor="black", alpha=0.8)
plt.title("Distribution of Trip Distance")
plt.xlabel("Distance (miles)")
plt.ylabel("Number of Trips")
plt.tight_layout()
plt.savefig("04_histogram.png", dpi=150)
plt.show()

# --- Box Plot: tip by pickup_borough ---
plt.figure(figsize=(9, 5))
df.boxplot(column="tip", by="pickup_borough", grid=False)
plt.title("Tip Amount by Pickup Borough")
plt.suptitle("")  # remove the automatic pandas super-title
plt.xlabel("Pickup Borough")
plt.ylabel("Tip ($)")
plt.tight_layout()
plt.savefig("05_box_plot.png", dpi=150)
plt.show()

# ---------------------------------------------------------------
# Visualizations using Seaborn
# ---------------------------------------------------------------

# --- Count Plot: trips per pickup_borough ---
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="pickup_borough", order=df["pickup_borough"].value_counts().index)
plt.title("Number of Trips by Pickup Borough")
plt.xlabel("Pickup Borough")
plt.ylabel("Count of Trips")
plt.tight_layout()
plt.savefig("06_count_plot.png", dpi=150)
plt.show()

# --- Scatter Plot: distance vs fare, coloured by pickup_borough ---
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="distance", y="fare", hue="pickup_borough", alpha=0.6)
plt.title("Distance vs Fare by Pickup Borough")
plt.xlabel("Distance (miles)")
plt.ylabel("Fare ($)")
plt.tight_layout()
plt.savefig("07_scatter_plot.png", dpi=150)
plt.show()

# --- Heatmap: correlation matrix ---
corr = df[["distance", "fare", "tip", "tolls", "total"]].corr()
plt.figure(figsize=(7, 5))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("08_heatmap.png", dpi=150)
plt.show()

# --- Pair Plot: distance, fare, tip, total coloured by pickup_zone ---
# There are ~190 pickup zones, which would make the plot unreadable,
# so only the 5 most frequent zones are used for the colouring.
top_zones = df["pickup_zone"].value_counts().head(5).index
pair_df = df[df["pickup_zone"].isin(top_zones)]

pair = sns.pairplot(
    pair_df,
    vars=["distance", "fare", "tip", "total"],
    hue="pickup_zone",
    palette="Set2",
    plot_kws={"alpha": 0.6},
)
pair.figure.suptitle("Pair Plot (Top 5 Pickup Zones)", y=1.02)
pair.savefig("09_pair_plot.png", dpi=150)
plt.show()

# --- Violin Plot: fare distribution per payment method ---
plt.figure(figsize=(8, 5))
sns.violinplot(data=df, x="payment", y="fare", hue="payment", palette="muted", legend=False)
plt.title("Fare Distribution by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Fare ($)")
plt.tight_layout()
plt.savefig("10_violin_plot.png", dpi=150)
plt.show()
