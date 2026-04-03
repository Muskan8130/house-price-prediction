from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(
    __name__,
    template_folder="../templates",
    static_folder="../static"
)

# Load trained pipeline (preprocessor + model)
model = joblib.load("models/house_price_model.pkl")
data = pd.read_excel("house_prices_v2.xlsx")

# clean columns
data.columns = data.columns.str.strip().str.lower()

# remove nulls
data = data.dropna(subset=["city", "locality"])

# convert to string safely
data["city"] = data["city"].astype(str).str.lower()
data["locality"] = data["locality"].astype(str).str.lower()

city_locality_map = {}

for city in data["city"].unique():
    city_locality_map[city] = list(
        data[data["city"] == city]["locality"].unique()
    )


def format_price(price):
    if price >= 10000000:
        return f"{price/10000000:.2f} Cr"
    elif price >= 100000:
        return f"{price/100000:.2f} Lakh"
    else:
        return str(price)

@app.route("/", methods=["GET", "POST"])
def home():
    formatted_price = None   # ✅ ADD
    low = None               # ✅ ADD
    high = None              # ✅ ADD

    if request.method == "POST":
        form_data = {
        "city": request.form["city"].lower(),
        "locality": request.form["locality"].lower(),
        "zone_category": request.form["zone_category"].lower(),
        "property_type": request.form["property_type"].lower(),
        "area_sqft": float(request.form["area_sqft"]),
        "bedrooms": int(request.form["bedrooms"]),
        "bathrooms": int(request.form["bathrooms"]),
        "balconies": int(request.form["balconies"]),
        "parking": int(request.form["parking"]),
        "floor_no": int(request.form["floor_no"]),
        "total_floors": int(request.form["total_floors"]),
        "age_of_property": int(request.form["age_of_property"]),
        "distance_from_metro_km": float(request.form["distance_from_metro_km"]),
        "furnishing_status": request.form["furnishing_status"].lower()
        }

        df = pd.DataFrame([form_data])
        raw_price = int(model.predict(df)[0])

        formatted_price = format_price(raw_price)
        low = format_price(int(raw_price * 0.9))
        high = format_price(int(raw_price * 1.1))

    return render_template(
        "index.html",
        price=formatted_price,
        low=low,
        high=high,
        city_locality_map=city_locality_map
    )


if __name__ == "__main__":
    app.run(debug=True)
