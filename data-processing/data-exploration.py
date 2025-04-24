import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv("bike_rental_data.csv")

df.hist(bins=50, figsize=(20, 15))
plt.savefig("report-use-img/hist.png")
plt.close()

plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.savefig("report-use-img/heatmap.png")
plt.close()

# no useful

# df.plot(kind="scatter", x="temp", y="humidity", alpha=0.3)
# plt.xlabel("Temperature vs Humidity")
# plt.savefig("report-use-img/humidity_scatter.png")
# plt.close()

# df.plot(kind="scatter", x="temp", y="windspeed", alpha=0.5)
# plt.title("Temperature vs Windspeed")
# plt.savefig("report-use-img/windspeed_scatter.png")
# plt.close()

df.plot(
    kind="scatter",
    x="temp",
    y="bikes_rented",
    alpha=0.4,
    s=df["temp"] * 10,
    figsize=(10, 7),
    c="bikes_rented",
    cmap=plt.get_cmap("jet"),
    colorbar=True,
)
plt.savefig("report-use-img/temp_bikes_rented.png")
plt.close()

plt.figure(figsize=(12, 7))
scatter = plt.scatter(
    df["hour"], df["temp"], c=df["bikes_rented"], cmap="jet", alpha=0.6, s=50
)

plt.colorbar(scatter, label="Bikes Rented")
plt.xlabel("Hour of Day")
plt.ylabel("Temperature")
plt.title("Bike Rentals by Hour and Temperature")
plt.xticks(range(0, 24, 2))
plt.grid(True, alpha=0.3)
plt.savefig("report-use-img/hour_temp_scatter.png")
plt.close()

plt.figure(figsize=(14, 6))

# 创建两个子图
plt.subplot(1, 2, 1)
# 筛选工作日数据
workday_data = df[df["workingday"] == 1]
# 按小时分组并计算平均租赁量
hourly_avg_workday = workday_data.groupby("hour")["bikes_rented"].mean()
plt.bar(hourly_avg_workday.index, hourly_avg_workday.values, color="blue", alpha=0.7)
plt.title("Average Bike Rentals by Hour (Working Days)")
plt.xlabel("Hour of Day")
plt.ylabel("Average Bikes Rented")
plt.xticks(range(0, 24, 2))
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
# 筛选非工作日数据
non_workday_data = df[df["workingday"] == 0]
# 按小时分组并计算平均租赁量
hourly_avg_non_workday = non_workday_data.groupby("hour")["bikes_rented"].mean()
plt.bar(
    hourly_avg_non_workday.index,
    hourly_avg_non_workday.values,
    color="green",
    alpha=0.7,
)
plt.title("Average Bike Rentals by Hour (Non-Working Days)")
plt.xlabel("Hour of Day")
plt.ylabel("Average Bikes Rented")
plt.xticks(range(0, 24, 2))
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("report-use-img/workingday_bar.png")
plt.close()

plt.figure(figsize=(12, 8))

# 为每个季节创建不同颜色的散点图
season_names = {1: "Winter", 2: "Spring", 3: "Summer", 4: "Autumn"}
colors = ["blue", "green", "red", "orange"]

for season, color in zip(range(1, 5), colors):
    season_data = df[df["season"] == season]
    plt.scatter(
        season_data["temp"],
        season_data["bikes_rented"],
        c=color,
        label=season_names[season],
        alpha=0.6,
        s=50,
    )

plt.xlabel("Temperature")
plt.ylabel("Bikes Rented")
plt.title("Relationship Between Temperature and Bike Rentals by Season")
plt.legend()
plt.grid(True, alpha=0.3)