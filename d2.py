import joblib
import pandas as pd

# Load the trained model
loaded_model = joblib.load("rf_price_predictor.pkl")
print("\n✅ Random Forest model loaded!")

# Load the dataset again (for testing)
df = pd.read_csv("D:\\codes_practice\\pylabmicro\\product_prices.csv")

# Clean and preprocess the dataset exactly like before
price_columns = ["farmprice", "atlantaretail", "chicagoretail", "losangelesretail", "newyorkretail"]
for col in price_columns:
    df[col] = df[col].replace(r'[\$,]', '', regex=True)
    df[col] = pd.to_numeric(df[col], errors='coerce')
df[price_columns] = df[price_columns].fillna(0)
df["averagespread"] = df["averagespread"].str.replace('%', '', regex=True).astype(float) / 100
df["profit_atlanta"] = df["atlantaretail"] - df["farmprice"]
df["profit_chicago"] = df["chicagoretail"] - df["farmprice"]
df["profit_losangeles"] = df["losangelesretail"] - df["farmprice"]
df["profit_newyork"] = df["newyorkretail"] - df["farmprice"]

# Prepare input data
features = ["farmprice", "profit_atlanta", "profit_chicago", "profit_losangeles", "profit_newyork"]
X = df[features]

# Predict on first 5 samples
sample_prediction = loaded_model.predict(X[:5])
print("\n🎯 Sample Predictions on Test Data:", sample_prediction)
