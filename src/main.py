import pandas as pd
import matplotlib
matplotlib.use("Agg")

# TODO : STEP 1) Dataset loaded
file_path = "house_prices_v2.xlsx"

df = pd.read_excel(file_path, engine="openpyxl")
"""
print("Dataset loaded successfully ✅")
print("Rows:", df.shape[0]) # 462 rows
print("Columns:", df.shape[1]) # 16 columns

print("\nColumn Names:")
print(df.columns)

print("\nFirst 5 rows:")
print(df.head())

"""

# TODO : STEP 2) Data check and clean

df.columns = df.columns.str.strip().str.lower()
df = df.drop("avg_price_per_sqft", axis=1)
print("Check spaces in column name and convert in lower casedf.columns : ",df.columns)
print("\nDuplicate rows:", df.duplicated().sum())
print("Check duplicate rows : " ,df.duplicated().sum())
print("Drop duplicated rows",df.drop_duplicates(inplace=True))
print(df.shape[0])
print(df.shape[1])
# ? Aear sqft should be not <=0
df = df[df["area_sqft"] > 0]
# ? Bathroom and Bedrooms should =>1
df = df[(df["bedrooms"] >= 1) &  (df["bathrooms"] >= 1)]
# ? age of property should not be negative
df.loc[df["age_of_property"] < 0 ,"age_of_property"] = df["age_of_property"].median()
# ? proce should be postive
df = df[df["price_total"] > 0]
# ? yha categorical cloumns ko string me convert , space remove and lower me convert kiya
categorical_cols = [
    "city",
    "locality",
    "zone_category",
    "property_type",
    "furnishing_status"
]

for col in categorical_cols:
    df[col] = df[col].astype(str).str.strip().str.lower()

# TODO : STEP 3)  Find X and y fetaure
X = df.drop("price_total",axis=1)
y = df["price_total"]
print(X.shape)
print(y.shape)

# TODO : STEP 4) Train / Test splits
from sklearn.model_selection import train_test_split
X_train, X_test , y_train , y_test = train_test_split(X,y , random_state = 42 , test_size = 0.2)
print(X_train.shape , X_test.shape)

# TODO : STEP 5) Piplining, OneHoteEncoding , Scalling
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# TODO : STEP 6) Linear Regression algo 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

numeric_features = [
    "area_sqft",
    "bedrooms",
    "bathrooms",
    "balconies",
    "parking",
    "floor_no",
    "total_floors",
    "age_of_property",
    "distance_from_metro_km"
]

categorical_features = [
    "city",
    "locality",
    "zone_category",
    "property_type",
    "furnishing_status"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

lr_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ]
)
scores = cross_val_score(
    lr_pipeline,
    X,
    y,
    cv=5,
    scoring="r2"
)

print("Linear Regression R2:", scores.mean())

# TODO : STEP 7) Random Forest algo

from sklearn.ensemble import RandomForestRegressor

rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        ))
    ]
)

rf_scores = cross_val_score(
    rf_pipeline,
    X,
    y,
    cv=5,
    scoring="r2"
)

print("Random Forest R2:", rf_scores.mean())

# TODO : STEP 8)  FINAL MODEL TRAINING (ON FULL DATA)

rf_pipeline.fit(X, y)
# ! matplotlip graph---------------------------------
import pandas as pd
import matplotlib.pyplot as plt

# Train final RF model on full training data
rf_pipeline.fit(X_train, y_train)

# Get feature names after preprocessing
feature_names = rf_pipeline.named_steps["preprocessor"].get_feature_names_out()

# Get importance
importances = rf_pipeline.named_steps["model"].feature_importances_

# Create DataFrame
feature_importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print(feature_importance_df.head(10))
plt.figure(figsize=(10,6))
plt.barh(
    feature_importance_df["feature"][:10],
    feature_importance_df["importance"][:10]
)
plt.gca().invert_yaxis()
plt.title("Top 10 Feature Importances")
plt.xlabel("Importance")
plt.show()

# ! matplotlip graph---------------------------------
# ! 7 EDA Graphs   ------------------------------
# 1 ) Price distribution
plt.figure(figsize=(8,5))
df["price_total"].plot(kind="hist", bins=30)
plt.title("House Price Distribution")
plt.xlabel("Price")
plt.show()
plt.close()
# 2) Area VS Price 
plt.figure(figsize=(8,5))
plt.scatter(df["area_sqft"], df["price_total"], alpha=0.5)
plt.xlabel("Area (sqft)")
plt.ylabel("Price")
plt.title("Area vs Price")
plt.show()
plt.close()

# 3) City wise avrage price 
df.groupby("city")["price_total"].mean().plot(kind="bar")
plt.title("Average Price by City")
plt.ylabel("Avg Price")
plt.show()
plt.close()

# 4) Zone category vs price
df.groupby("zone_category")["price_total"].mean().plot(kind="bar")
plt.title("Zone Category vs Price")
plt.show()
plt.close()

# 5) Bedrooms vs price 
df.groupby("bedrooms")["price_total"].mean().plot(kind="line", marker="o")
plt.title("Bedrooms vs Price")
plt.show()
plt.close()

# 6) Age vs price
plt.scatter(df["age_of_property"], df["price_total"], alpha=0.5)
plt.xlabel("Age of Property")
plt.ylabel("Price")
plt.title("Age vs Price")
plt.show()
plt.close()

# 7) Distance from metro vs price
plt.scatter(df["distance_from_metro_km"], df["price_total"], alpha=0.5)
plt.xlabel("Distance from Metro (km)")
plt.ylabel("Price")
plt.title("Metro Distance vs Price")
plt.show()
plt.close()


# ! 7 EDA Graphs -----------------------------------
# TODO : STEP 9) Feature importance

import pandas as pd

feature_names = rf_pipeline.named_steps["preprocessor"].get_feature_names_out()
importances = rf_pipeline.named_steps["model"].feature_importances_

feat_imp = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print(feat_imp.head(10))


# TODO : STEP 10) Error Analsys
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

y_pred = rf_pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("MAE:", mae)
print("RMSE:", rmse)


# TODO : STEP 11) User input price predcition
user_input = pd.DataFrame([{
    "city": "noida",
    "locality": "sector 62",
    "zone_category": "city",
    "property_type": "apartment",
    "area_sqft": 1500,
    "bedrooms": 3,
    "bathrooms": 3,
    "balconies": 2,
    "parking": 1,
    "floor_no": 8,
    "total_floors": 18,
    "age_of_property": 6,
    "distance_from_metro_km": 1.8,
    "furnishing_status": "semi-furnished"
}])

predicted_price = rf_pipeline.predict(user_input)
print("Predicted House Price:", int(predicted_price[0]))


# TODO : STEP 12) Save final trained pipeline
import joblib
import os

os.makedirs("models", exist_ok=True)

joblib.dump(rf_pipeline, "models/house_price_model.pkl")

print("✅ Model saved successfully at models/house_price_model.pkl")





