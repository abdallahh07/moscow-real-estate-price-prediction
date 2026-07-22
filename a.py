import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor,VotingRegressor,StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder , PolynomialFeatures 
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score,RepeatedKFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import shap
import gc
from sklearn import set_config
set_config(transform_output="pandas") 

### 1. load the files
d = pd.read_csv("district_prices_monthly.csv")
m = pd.read_csv("metro_stations.csv")
n = pd.read_csv("new_builds.csv")
r = pd.read_csv("rentals.csv")
s = pd.read_csv("secondary_market.csv")

### 2. review the data
#print(d.head())
#print(d.columns)
#print(d.dtypes)
#print(m.head())
print(n.head())
print(n.columns)
#print(r.head())
#print(s.head())

### 3. analyze the data(EDA)
  # 3.1 analyze the district prices monthly 
def analyze_districts_prices():
  print (pd.DataFrame(d.describe()))
  plt.figure(figsize=(10,6))
  sns.heatmap(data=d.select_dtypes(include = "number").corr(),
              cmap="magma",
              annot=True)
  plt.title("District prices map")
  plt.show()
  
  fig,ax = plt.subplots(3,3,figsize=(20,15))
  ax = ax.flatten()
  
  # 3.1.1 district prices 
  district_prices = d.groupby("district")["newbuild_price_per_sqm"].mean().sort_values(ascending=False).head(10)
  sns.barplot(x = district_prices.index,
              y = district_prices.values,
              palette = "magma",
              ax=ax[0])
  ax[0].set_xticklabels(ax[0].get_xticklabels(),fontsize=6)
  ax[0].set_title("district prices")
  
  # 3.1.2 rental prices by district
  rental_prices = d.groupby("district")["rental_price_per_sqm_monthly"].mean().sort_values(ascending=False).head(10)
  sns.barplot(x = rental_prices.index,
              y=rental_prices.values,
              palette="magma",
              ax=ax[1])
  ax[1].set_xticklabels(ax[1].get_xticklabels(),fontsize=6)
  ax[1].set_title("Rental Prices by districts")
  
  # 3.1.3 average mortage rate ditribution
  sns.kdeplot(d["avg_mortgage_rate_pct"].dropna(),
              color='#6b0f8e',
              fill=True,
              ax=ax[2])
  ax[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.1f}%'))
  ax[2].set_title("Average mortage rate distribution")
  
  # 3.1.4 CBR key rates vs avg mortage rate over time
  d["year_month"] = pd.to_datetime(d["year_month"])
  rates_over_time = d.groupby("year_month")[["cbr_key_rate_pct", "avg_mortgage_rate_pct"]].mean()
  ax[3].plot(rates_over_time.index,rates_over_time["cbr_key_rate_pct"],
          color='#6b0f8e',linewidth = 2 ,label = "CBR Key Rate",linestyle = '--')
  ax[3].plot(rates_over_time.index,rates_over_time["avg_mortgage_rate_pct"],
          color='#c0395a',linewidth = 2 ,label = "Avg Mortgage Rate",linestyle = '-')
  ax[3].set_title("CBR Key Rate vs Avg Mortgage Rate Over Time")
  ax[3].set_ylabel("Rate %")  
  ax[3].legend(loc='upper left', fontsize=8)
  
  # 3.1.5 Price per SQM n
  realestat = d.groupby("year_month")[["newbuild_price_per_sqm","secondary_price_per_sqm"]].mean()
  ax[4].plot(realestat.index,realestat["newbuild_price_per_sqm"],
             color ='#6b0f8e', linewidth= 2 ,label = "New build price",linestyle = "--")
  ax[4].plot(realestat.index,realestat["secondary_price_per_sqm"],
             color = '#c0395a', linewidth = 2 , label = "Secondary Price",linestyle = "-")
  ax[4].set_title("Real estate overview")
  ax[4].set_ylabel("Price per SQM")
  ax[4].legend(loc='upper left', fontsize=8)
  
  # 3.1.6 the real-estate overview by number of units
  listing = d.groupby("year_month")[["n_listings_secondary", "n_listings_newbuild", "n_listings_rental"]].mean()
  ax[5].plot(listing.index,listing["n_listings_secondary"],
             color = '#6b0f8e',linewidth = 2 , label = "Secondary Market",linestyle = '-')
  ax[5].plot(listing.index,listing["n_listings_newbuild"],
             color = '#c0395a', linewidth = 2 , label = "Newbuild Market",linestyle = '--')
  ax[5].plot(listing.index,listing["n_listings_rental"],
             color = '#ff8c00' ,linewidth = 2 , label = "Rental listing", linestyle = ':')
  ax[5].set_title("Listings Over Time by Type")
  ax[5].set_ylabel("Number of units")
  ax[5].legend(loc='upper right', fontsize=8)
  
  # 3.1.7 Rental yield estimate
  d["rental_yield"] = (d["rental_price_per_sqm_monthly"] * 12) / d["secondary_price_per_sqm"]
  yield_over_time = d.groupby("year_month")["rental_yield"].mean()  
  yield_over_time = d.groupby("year_month")["rental_yield"].mean().reset_index()
  ax[6].plot(yield_over_time["year_month"], yield_over_time["rental_yield"],
            color='#6b0f8e', linewidth=2, label="Rental Yield")
  ax[6].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.3f}%'))
  ax[6].set_title("Rental yield")
  
  # 3.1.8  Price per Okrug
  okrug = d.groupby("okrug")["newbuild_price_per_sqm"].mean().sort_values(ascending=False).head(10)
  sns.barplot(x = okrug.index,
              y = okrug.values,
              palette="magma",
              ax=ax[7])
  ax[7].set_title("Price by Okrug")
  
  # 3.1.9 Relation between Mortage and price
  sns.boxplot(x="avg_mortgage_rate_pct", y="secondary_price_per_sqm",
            data=d,
            ax=ax[8], palette="magma",
            hue="avg_mortgage_rate_pct", legend=False)
  ax[8].set_title("Secondary Price Distribution by Mortgage Rate")
  ax[8].set_xlabel("Avg Mortgage Rate %")
  ax[8].set_ylabel("Price per SQM")
  ax[8].tick_params(axis='x', rotation=45)  
  
  
  plt.show()
#analyze_districts_prices()
print(m["lat"].value_counts().sum())

  # 3.2 Analyzing metro
def analyze_metro():
  fig,ax = plt.subplots(2,3,figsize=(20,12))
  ax = ax.flatten()

  # 3.2.1 Numbers of station names
  stations_per_line = m["line"].value_counts()
  sns.barplot(x=stations_per_line.values, y=stations_per_line.index,
              palette="magma", hue=stations_per_line.index, legend=False,
              ax=ax[0])
  ax[0].set_title("Stations per Metro Line")
  ax[0].set_xlabel("Number of Stations")
  
  # 3.2.2 distance to center distribution 
  sns.kdeplot(m["to_center_km"].dropna(),
            ax=ax[1], color='#6b0f8e', fill=True)
  ax[1].set_title("Distance to Center Distribution")
  ax[1].set_xlabel("Distance (km)")
  ax[1].set_ylabel("Density")
  ax[1].axvline(m["to_center_km"].median(), color='#c0395a', 
              linestyle='--', linewidth=1.5,
              label=f'Median: {m["to_center_km"].median():.1f} km')
  ax[1].legend()

  # 3.2.3 Metro lines opened by year
  m["decade"] = (m["year_opened"].astype(int) // 10) * 10
  decade_counts = m["decade"].value_counts().sort_index()
  sns.barplot(x=decade_counts.index, y=decade_counts.values,
              palette="magma", hue=decade_counts.index, legend=False,
              ax=ax[2])
  ax[2].set_title("Stations Opened by Decade")
  ax[2].set_xlabel("Decade")
  ax[2].set_ylabel("Count")

  # 3.2.4 Distance to center by line
  sns.boxplot(x="to_center_km", y="line", data=m,
            palette="magma", hue="line", legend=False,
            ax=ax[3])
  ax[3].set_title("Distance to Center by Metro Line")
  ax[3].set_xlabel("Distance (km)")
  ax[3].set_ylabel("Line")
  ax[3].tick_params(axis='y', labelsize=7)

  # 3.2.5 Rental Price vs Metro Distance
  ax[4].scatter(r["metro_distance_min"], r["monthly_rent_rub"],
                color='#6b0f8e', alpha=0.2, s=10)
  ax[4].set_title("Rental Price vs Metro Distance")
  ax[4].set_xlabel("Distance (min)")
  ax[4].set_ylabel("Monthly Rent (RUB)")

  # 3.2.6 Secondary Market Price vs Metro Distance
  ax[5].scatter(s["metro_distance_min"], s["price_rub"],
              color='#6b0f8e', alpha=0.2, s=10)
  ax[5].set_title("Secondary Price vs Distance to Center")
  ax[5].set_xlabel("Distance (min)")
  ax[5].set_ylabel("Price (RUB)")

  plt.show()
  
  # 3.3 Analyze new builds 
def analyze_new_builds():
  n["date_posted"] = pd.to_datetime(n["date_posted"], errors="coerce")
  n["price_rub"] = pd.to_numeric(n["price_rub"], errors="coerce")
  n["price_per_sqm"] = pd.to_numeric(n["price_per_sqm"], errors="coerce")
  n["total_area"] = pd.to_numeric(n["total_area"], errors="coerce")
  n["to_center_km"] = pd.to_numeric(n["to_center_km"], errors="coerce")
  n["mortgage_rate_at_listing"] = pd.to_numeric(n["mortgage_rate_at_listing"], errors="coerce")

  fig, axes = plt.subplots(3, 3, figsize=(20, 15))
  ax = axes.flatten()
  fig.suptitle("New Builds Analysis", fontsize=16, fontweight='bold', y=1.01)

  # 3.3.1 Complex class distribution
  class_counts = n["complex_class"].value_counts()
  sns.barplot(x=class_counts.index, y=class_counts.values,
              palette="magma", hue=class_counts.index, legend=False, ax=ax[0])
  ax[0].set_title("Complex Class Distribution")
  ax[0].set_ylabel("Count")

  # 3.3.2 Price per sqm by complex class
  sns.boxplot(x="complex_class", y="price_per_sqm", data=n,
              palette="magma", hue="complex_class", legend=False, ax=ax[1])
  ax[1].set_title("Price per SQM by Complex Class")
  ax[1].set_xlabel("Class")
  ax[1].set_ylabel("Price per SQM")

  # 3.3.3 Top 10 developers by listings
  top_devs = n["developer"].value_counts().head(10)
  sns.barplot(x=top_devs.values, y=top_devs.index,
              palette="magma", hue=top_devs.index, legend=False, ax=ax[2])
  ax[2].set_title("Top 10 Developers by Listings")
  ax[2].set_xlabel("Count")

  # 3.3.4 Ready status distribution
  status_counts = n["ready_status"].value_counts()
  sns.barplot(x=status_counts.values, y=status_counts.index,
              palette="magma", hue=status_counts.index, legend=False, ax=ax[3])
  ax[3].set_title("Ready Status Distribution")
  ax[3].set_xlabel("Count")

  # 3.3.5 Price per sqm by okrug
  okrug_price = n.groupby("okrug")["price_per_sqm"].mean().sort_values(ascending=False)
  sns.barplot(x=okrug_price.values, y=okrug_price.index,
              palette="magma", hue=okrug_price.index, legend=False, ax=ax[4])
  ax[4].set_title("Avg Price per SQM by Okrug")
  ax[4].set_xlabel("Price per SQM")

  # 3.3.6 Price vs total area scatter
  ax[5].scatter(n["total_area"], n["price_rub"],
                color='#6b0f8e', alpha=0.2, s=10)
  ax[5].set_title("Price vs Total Area")
  ax[5].set_xlabel("Total Area (sqm)")
  ax[5].set_ylabel("Price (RUB)")

  # 3.3.7 Mortgage rate distribution
  sns.kdeplot(n["mortgage_rate_at_listing"].dropna(),
              color='#6b0f8e', fill=True, ax=ax[6])
  ax[6].set_title("Mortgage Rate Distribution")
  ax[6].set_xlabel("Mortgage Rate %")

  # 3.3.8 Price vs metro distance
  ax[7].scatter(n["metro_distance_min"], n["price_rub"],
                color='#6b0f8e', alpha=0.2, s=10)
  ax[7].set_title("Price vs Metro Distance")
  ax[7].set_xlabel("Distance (min)")
  ax[7].set_ylabel("Price (RUB)")

  # 3.3.9 Deal type distribution
  deal_counts = n["deal_type"].value_counts()
  ax[8].pie(deal_counts.values, labels=deal_counts.index,
            autopct='%1.1f%%', colors=['#3b0f70', '#c0395a', '#f1605d'],
            wedgeprops={"edgecolor": "white"})
  ax[8].set_title("Deal Type Distribution")

  plt.tight_layout()
  plt.show()

def analyze_rentals():
  r["date_posted"] = pd.to_datetime(r["date_posted"], errors="coerce")
  r["monthly_rent_rub"] = pd.to_numeric(r["monthly_rent_rub"], errors="coerce")
  r["rent_per_sqm"] = pd.to_numeric(r["rent_per_sqm"], errors="coerce")
  r["total_area"] = pd.to_numeric(r["total_area"], errors="coerce")
  r["to_center_km"] = pd.to_numeric(r["to_center_km"], errors="coerce")

  fig, axes = plt.subplots(3, 3, figsize=(20, 15))
  ax = axes.flatten()
  fig.suptitle("Rentals Analysis", fontsize=16, fontweight='bold', y=1.01)

  # 3.4.1 Rent by okrug
  okrug_rent = r.groupby("okrug")["monthly_rent_rub"].mean().sort_values(ascending=False)
  sns.barplot(x=okrug_rent.values, y=okrug_rent.index,
              palette="magma", hue=okrug_rent.index, legend=False, ax=ax[0])
  ax[0].set_title("Avg Rent by Okrug")
  ax[0].set_xlabel("Monthly Rent (RUB)")

  # 3.4.2 Rent by number of rooms
  sns.boxplot(x="rooms", y="monthly_rent_rub", data=r,
              palette="magma", hue="rooms", legend=False, ax=ax[1])
  ax[1].set_title("Rent by Number of Rooms")
  ax[1].set_xlabel("Rooms")
  ax[1].set_ylabel("Monthly Rent (RUB)")

  # 3.4.3 Building type distribution
  building_counts = r["building_type"].value_counts()
  sns.barplot(x=building_counts.values, y=building_counts.index,
              palette="magma", hue=building_counts.index, legend=False, ax=ax[2])
  ax[2].set_title("Building Type Distribution")
  ax[2].set_xlabel("Count")

  #3.4.4 Pets allowed vs rent
  sns.boxplot(x="pets_allowed", y="monthly_rent_rub", data=r,
              palette="magma", hue="pets_allowed", legend=False, ax=ax[3])
  ax[3].set_title("Pets Allowed vs Rent")
  ax[3].set_xlabel("Pets Allowed")
  ax[3].set_ylabel("Monthly Rent (RUB)")

  #3.4.5 Rent vs total area
  ax[4].scatter(r["total_area"], r["monthly_rent_rub"],
                color='#6b0f8e', alpha=0.2, s=10)
  ax[4].set_title("Rent vs Total Area")
  ax[4].set_xlabel("Total Area (sqm)")
  ax[4].set_ylabel("Monthly Rent (RUB)")

  #3.4.6 Renovation type vs rent
  sns.boxplot(x="renovation", y="monthly_rent_rub", data=r,
              palette="magma", hue="renovation", legend=False, ax=ax[5])
  ax[5].set_title("Renovation Type vs Rent")
  ax[5].set_xlabel("Renovation")
  ax[5].set_ylabel("Monthly Rent (RUB)")
  ax[5].tick_params(axis='x', rotation=45)

  #3.4.7 Rent distribution
  sns.kdeplot(r["monthly_rent_rub"].dropna(),
              color='#6b0f8e', fill=True, ax=ax[6])
  ax[6].set_title("Rent Distribution")
  ax[6].set_xlabel("Monthly Rent (RUB)")
  ax[6].axvline(r["monthly_rent_rub"].median(), color='#c0395a',
                linestyle='--', linewidth=1.5,
                label=f'Median: {r["monthly_rent_rub"].median():,.0f}')
  ax[6].legend()


  # 3.4.8 Rent vs metro distance
  ax[7].scatter(r["metro_distance_min"], r["monthly_rent_rub"],
                color='#6b0f8e', alpha=0.2, s=10)
  ax[7].set_title("Rent vs Metro Distance")
  ax[7].set_xlabel("Distance (min)")
  ax[7].set_ylabel("Monthly Rent (RUB)")

  #3.4.9 Furnished vs rent
  sns.boxplot(x="furnished", y="monthly_rent_rub", data=r,
              palette="magma", hue="furnished", legend=False, ax=ax[8])
  ax[8].set_title("Furnished vs Rent")
  ax[8].set_xlabel("Furnished")
  ax[8].set_ylabel("Monthly Rent (RUB)")

  plt.tight_layout()
  plt.show()

  # 3.5 Analyze new builds
def analyze_secondary():
    s["date_posted"] = pd.to_datetime(s["date_posted"], errors="coerce")
    s["price_rub"] = pd.to_numeric(s["price_rub"], errors="coerce")
    s["price_per_sqm"] = pd.to_numeric(s["price_per_sqm"], errors="coerce")
    s["total_area"] = pd.to_numeric(s["total_area"], errors="coerce")
    s["to_center_km"] = pd.to_numeric(s["to_center_km"], errors="coerce")

    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    ax = axes.flatten()
    fig.suptitle("Secondary Market Analysis", fontsize=16, fontweight='bold', y=1.01)

    # 3.5.1 Price per sqm by okrug
    okrug_price = s.groupby("okrug")["price_per_sqm"].mean().sort_values(ascending=False)
    sns.barplot(x=okrug_price.values, y=okrug_price.index,
                palette="magma", hue=okrug_price.index, legend=False, ax=ax[0])
    ax[0].set_title("Avg Price per SQM by Okrug")
    ax[0].set_xlabel("Price per SQM")

    # 3.5.2 Price by number of rooms
    sns.boxplot(x="rooms", y="price_rub", data=s,
                palette="magma", hue="rooms", legend=False, ax=ax[1])
    ax[1].set_title("Price by Number of Rooms")
    ax[1].set_xlabel("Rooms")
    ax[1].set_ylabel("Price (RUB)")

    # 3.5.3 Building type distribution
    building_counts = s["building_type"].value_counts()
    sns.barplot(x=building_counts.values, y=building_counts.index,
                palette="magma", hue=building_counts.index, legend=False, ax=ax[2])
    ax[2].set_title("Building Type Distribution")
    ax[2].set_xlabel("Count")

    # 3.5.4 Renovation type vs price
    sns.boxplot(x="renovation", y="price_per_sqm", data=s,
                palette="magma", hue="renovation", legend=False, ax=ax[3])
    ax[3].set_title("Renovation Type vs Price per SQM")
    ax[3].set_xlabel("Renovation")
    ax[3].tick_params(axis='x', rotation=45)

    # 3.5.5 Price vs total area
    ax[4].scatter(s["total_area"], s["price_rub"],
                  color='#6b0f8e', alpha=0.2, s=10)
    ax[4].set_title("Price vs Total Area")
    ax[4].set_xlabel("Total Area (sqm)")
    ax[4].set_ylabel("Price (RUB)")

    # 3.5.6 Seller type distribution
    seller_counts = s["seller_type"].value_counts()
    ax[5].pie(seller_counts.values, labels=seller_counts.index,
              autopct='%1.1f%%', colors=['#3b0f70', '#c0395a'],
              wedgeprops={"edgecolor": "white"})
    ax[5].set_title("Seller Type Distribution")

    # 3.5.7 Price distribution
    sns.kdeplot(s["price_rub"].dropna(),
                color='#6b0f8e', fill=True, ax=ax[6])
    ax[6].set_title("Price Distribution")
    ax[6].set_xlabel("Price (RUB)")
    ax[6].axvline(s["price_rub"].median(), color='#c0395a',
                  linestyle='--', linewidth=1.5,
                  label=f'Median: {s["price_rub"].median():,.0f}')
    ax[6].legend()

    # 3.5.8 Metro distance type vs price
    sns.boxplot(x="metro_distance_type", y="price_per_sqm", data=s,
                palette="magma", hue="metro_distance_type", legend=False, ax=ax[7])
    ax[7].set_title("Walk vs Transport Metro vs Price")
    ax[7].set_xlabel("Metro Access Type")
    ax[7].set_ylabel("Price per SQM")

    # 3.5.9 Price vs metro distance
    ax[8].scatter(s["metro_distance_min"], s["price_rub"],
                  color='#6b0f8e', alpha=0.2, s=10)
    ax[8].set_title("Price vs Metro Distance")
    ax[8].set_xlabel("Distance (min)")
    ax[8].set_ylabel("Price (RUB)")

    plt.tight_layout()
    plt.show()

#analyze_secondary()

### 4. Data cleaning and feature engineering
n["date_posted"] = pd.to_datetime(n["date_posted"], errors="coerce")
n["years_to_completion"] = n["completion_year"] - n["date_posted"].dt.year
n["year_posted"] = n["date_posted"].dt.year
n["month_posted"] = n["date_posted"].dt.month

### 5. Preprocessing
n_trans = ["developer", "complex_class", "district",
           "okrug", "ready_status", "metro_station", "metro_line", "deal_type"]

x = n.drop(["price_rub", "price_per_sqm", "id", "date_posted",
             "complex_id", "complex_name"], axis=1)
y = n["price_rub"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

### 6. Model Evaluation
models = {
    "RandomForestRegressor": RandomForestRegressor(),
    "GradientBoostingRegressor": GradientBoostingRegressor(),
    "LinearRegression": LinearRegression(),
    "KNeighborsRegressor": KNeighborsRegressor(),
    "XGBRegressor": XGBRegressor(),
    "CatBoostRegressor": CatBoostRegressor(verbose=0),
    "LGBMRegressor": LGBMRegressor(),
    "Ridge": Ridge(),
    "Lasso": Lasso()}

result = []
best_pipe = None
best_model = None
best_r2 = -999
meanabsoluteerror = 0
meansquarederror = 0

for model, mod in models.items():
    pipe = Pipeline([
        ("encoder", ColumnTransformer([
            ("one_hot", OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
            n_trans)], remainder="passthrough")),
        ("scaler", StandardScaler(with_mean=False)),
        ("model", mod)])
    pipe.fit(x_train, y_train)
    y_pred = pipe.predict(x_test)

    r2 = r2_score(y_test, y_pred)
    mean = mean_absolute_error(y_test, y_pred)
    mean_squared = mean_squared_error(y_test, y_pred)

    print(f"-------{model}-------")
    print("R2", r2)
    print("Mean", mean)
    print("Mean_squared", mean_squared)

    result.append({
        "Name": model,
        "Model": mod,
        "Pipe": pipe,
        "R2": r2,
        "Mean_squared_error": mean,
        "Mean_squared": mean_squared})

    if r2 > best_r2:
        best_pipe = pipe
        best_model = mod
        best_r2 = r2
        meanabsoluteerror = mean
        meansquarederror = mean_squared

### 7. Print best model
print("-------New_Builds-------")
print(f"Best Model is : {best_model}")
print(f"R2: {best_r2}")
print(f"Mean absolute error: {meanabsoluteerror}")
print(f"Mean squared error : {meansquarederror}")

### 8. Cross validation
rkf = RepeatedKFold(n_splits=5,n_repeats=3,random_state=42)
cv_scores = cross_val_score(best_pipe, x, y, cv=rkf, scoring="r2")
print(f"CV R2: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

### 9. ensemble
voting = VotingRegressor([
  ("Catboost",CatBoostRegressor(verbose=0)),
  ("lgbm",LGBMRegressor()),
  ("XGB",XGBRegressor())])

pipe_voting = Pipeline([
  ("encoder",ColumnTransformer([
    ("one_hot",OneHotEncoder(sparse_output=False,handle_unknown="ignore"),
     n_trans)],remainder="passthrough")),
  ("scaler",StandardScaler(with_mean=False)),
  ("model",voting)])

stacking = StackingRegressor(
  estimators=[
    ("catboost",CatBoostRegressor(verbose=0)),
    ("lgbm",LGBMRegressor()),
    ("xgb",XGBRegressor()),
    ("rf",RandomForestRegressor())],final_estimator=Ridge())

pipe_stacking = Pipeline([
  ("encoder",ColumnTransformer([
  ("one_hot",OneHotEncoder(sparse_output=False,handle_unknown="ignore"),
   n_trans)],remainder = "passthrough")),
  ("scaler",StandardScaler(with_mean=False)),
  ("model",stacking)]) 

for name,pipe in [("Voting",pipe_voting),("stacking",pipe_stacking)]:
  pipe.fit(x_train,y_train)
  y_preds = pipe.predict(x_test)
  r2 = r2_score(y_test, y_preds)
  cv = cross_val_score(pipe,x,y,cv=rkf,scoring="r2")
  print(f"-------{name}-------")
  print(f"R2: {r2:.4f}")
  print(f"CV R2: {cv.mean():.4f} ± {cv.std():.4f}")
  
y_pred = best_pipe.predict(x_test)
### 10. Model analysis
# 10.1 Result table
result_table = pd.DataFrame(result).sort_values("R2",ascending=False)

# 10.2 Residual plot
res = y_test - y_pred
plt.scatter(y_pred,res,
            alpha=0.3,)

plt.title("Residual plot")
plt.show()

# 10.3 Actual vs predicted
plt.scatter(y_test,y_pred,alpha=0.3)
plt.plot([y_test.min(),y_test.max()],
         [y_test.min(),y_test.max()],"r--")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.show()

# 10.4 feature importance
feat_imp = best_model.get_feature_importance()
feat_names = best_pipe.named_steps["encoder"].get_feature_names_out()
feat_df= pd.DataFrame({"Features":feat_names,"importance":feat_imp})
feat_df= feat_df.sort_values("importance",ascending=False).head(10)
sns.barplot(x="importance", y="Features", data=feat_df)
plt.title("Feature Importance")
plt.show()

# 10.5 Shap
x_test_transformed = best_pipe[:-1].transform(x_test)
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(x_test_transformed)
shap.summary_plot(shap_values, x_test_transformed)