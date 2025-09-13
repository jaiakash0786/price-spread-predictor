import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import joblib
import os
import sys
from datetime import datetime

# ==========================
# STEP 1: Load and Clean Dataset
# ==========================

try:
    df = pd.read_csv("D:\\codes_practice\\pylabmicro\\product_prices.csv")
except FileNotFoundError:
    print("❗ Error: Dataset file not found. Please check the file path.")
    sys.exit(1)
except Exception as e:
    print(f"❗ Error while reading the dataset: {str(e)}")
    sys.exit(1)

try:
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

except Exception as e:
    print(f"❗ Error during data cleaning: {str(e)}")
    sys.exit(1)

# ==========================
# STEP 2: Model Training
# ==========================

try:
    features = ["farmprice", "profit_atlanta", "profit_chicago", "profit_losangeles", "profit_newyork"]
    target = "averagespread"

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    rf_accuracy = rf_model.score(X_test, y_test)
    y_pred = rf_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"\n🌟 Random Forest Model Accuracy: {rf_accuracy:.2%}")
    print("\n📊 Model Evaluation Metrics:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.2%}")

    joblib.dump(rf_model, "rf_price_predictor.pkl")
    print("\n✅ Model saved successfully as rf_price_predictor.pkl")

except Exception as e:
    print(f"❗ Error during model training: {str(e)}")
    sys.exit(1)

# ==========================
# STEP 3: Data Visualizations
# ==========================

try:
    plt.figure(figsize=(8, 5))
    plt.scatter(df["farmprice"], df["averagespread"], color='green', alpha=0.6)
    plt.title("Farm Price vs Average Spread")
    plt.xlabel("Farm Price ($)")
    plt.ylabel("Average Spread")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
    plt.title("Actual vs Predicted Average Spread")
    plt.xlabel("Actual Spread")
    plt.ylabel("Predicted Spread")
    plt.grid(True)
    plt.show()

    avg_profits = [
        df["profit_atlanta"].mean(),
        df["profit_chicago"].mean(),
        df["profit_losangeles"].mean(),
        df["profit_newyork"].mean()
    ]

    cities = ["Atlanta", "Chicago", "Los Angeles", "New York"]

    plt.figure(figsize=(8, 5))
    plt.bar(cities, avg_profits, color=['orange', 'blue', 'green', 'purple'])
    plt.title("Average Profit Margin per City")
    plt.xlabel("City")
    plt.ylabel("Average Profit ($)")
    plt.grid(True, axis='y')
    plt.show()

except Exception as e:
    print(f"❗ Error during data visualization: {str(e)}")


# ==========================
# STEP 4: Prediction & Report Export Function
# ==========================

def export_report(city, product, quantity, predicted_spread, best_quantity, max_profit, actual_loss):
    try:
        report_content = f"""
========== PRODUCT PRICE SPREAD PREDICTION REPORT ==========

📍 City                : {city.title()}
🌾 Product             : {product.title()}
📦 Quantity Requested  : {quantity}

📊 Predicted Spread    : {predicted_spread:.2%}
✅ Suggested Quantity  : {best_quantity}
💰 Expected Max Profit : ${max_profit:.2f}
❗ Loss without Model  : ${actual_loss:.2f}

🕒 Report Generated On : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
============================================================
"""

        report_filename = f"prediction_report_{product.lower()}_{city.lower()}.txt"
        
        with open(report_filename, "w", encoding="utf-8") as file:
            file.write(report_content)


        print(f"\n📄 Report successfully saved as {report_filename}")
    except Exception as e:
        print(f"❗ Error while exporting report: {str(e)}")


def predict_best_quantity(city, product, quantity):
    try:
        if not os.path.exists("rf_price_predictor.pkl"):
            return "❗ Error: Model file not found!"

        model = joblib.load("rf_price_predictor.pkl")

        city_column_map = {
            "atlanta": "profit_atlanta",
            "chicago": "profit_chicago",
            "losangeles": "profit_losangeles",
            "newyork": "profit_newyork"
        }

        if city.lower() not in city_column_map:
            return "❗ Error: Invalid city name!"

        profit_column = city_column_map[city.lower()]
        product_data = df[df["productname"].str.lower() == product.lower()]

        if product_data.empty:
            return "❗ Error: Product not found!"

        if quantity <= 0:
            return "❗ Error: Quantity must be positive."

        avg_farmprice = product_data["farmprice"].mean()
        avg_profit_atlanta = product_data["profit_atlanta"].mean()
        avg_profit_chicago = product_data["profit_chicago"].mean()
        avg_profit_losangeles = product_data["profit_losangeles"].mean()
        avg_profit_newyork = product_data["profit_newyork"].mean()

        input_data = [[
            avg_farmprice,
            avg_profit_atlanta,
            avg_profit_chicago,
            avg_profit_losangeles,
            avg_profit_newyork
        ]]

        predicted_spread = model.predict(input_data)[0]
        avg_profit_per_unit = product_data[profit_column].mean()

        max_profit_amount = quantity * avg_profit_per_unit * predicted_spread
        best_quantity = int((quantity * predicted_spread) / (predicted_spread + 1))
        actual_loss = (quantity - best_quantity) * avg_profit_per_unit * (1 - predicted_spread)

        # --- Plots ---
        quantities = list(range(1, quantity + 1))
        profits = [q * avg_profit_per_unit * predicted_spread for q in quantities]

        plt.figure(figsize=(8, 5))
        plt.plot(quantities, profits, color='green')
        plt.axvline(best_quantity, color='red', linestyle='--', label=f'Best Quantity = {best_quantity}')
        plt.title(f"Profit vs Quantity for {product.title()} in {city.title()}")
        plt.xlabel("Quantity")
        plt.ylabel("Expected Profit ($)")
        plt.legend()
        plt.grid(True)
        plt.show()

        losses_full = [(quantity - q) * avg_profit_per_unit * (1 - predicted_spread) for q in quantities]
        losses_best = [(best_quantity - q) * avg_profit_per_unit * (1 - predicted_spread)
                       if q <= best_quantity else 0 for q in quantities]
        remaining_stock = [quantity - q for q in quantities]

        plt.figure(figsize=(8, 5))
        plt.plot(quantities, losses_full, color='orange', label='Loss (Full Quantity)')
        plt.plot(quantities, losses_best, color='blue', linestyle='--', label='Loss (Best Quantity)')
        plt.plot(quantities, remaining_stock, color='purple', label='Remaining Stock')
        plt.title(f"Loss & Remaining Stock vs Quantity for {product.title()} in {city.title()}")
        plt.xlabel("Quantity Purchased")
        plt.ylabel("Loss / Remaining Stock")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Export report
        export_report(city, product, quantity, predicted_spread, best_quantity, max_profit_amount, actual_loss)

        return f"""
✅ Prediction Result:
📍 City: {city.title()}
🌾 Product: {product.title()}
📊 Predicted Spread: {predicted_spread:.2%}
💰 Suggested Quantity to Buy: {best_quantity} out of {quantity} units
🏆 Expected Maximum Profit: ${max_profit_amount:.2f}
❗ Loss if you buy full quantity ({quantity} units): ${actual_loss:.2f}
"""

    except Exception as e:
        return f"❗ Error in prediction: {str(e)}"


def plot_quantity_vs_profit(city, product):
    try:
        city_column_map = {
            "atlanta": "profit_atlanta",
            "chicago": "profit_chicago",
            "losangeles": "profit_losangeles",
            "newyork": "profit_newyork"
        }

        if city.lower() not in city_column_map:
            print("❗ Invalid city name!")
            return

        profit_column = city_column_map[city.lower()]
        product_data = df[df["productname"].str.lower() == product.lower()]

        if product_data.empty:
            print("❗ Product not found!")
            return

        avg_profit_per_unit = product_data[profit_column].mean()
        quantities = list(range(1, 101))
        profits = [q * avg_profit_per_unit for q in quantities]

        plt.figure(figsize=(8, 5))
        plt.plot(quantities, profits, color='red')
        plt.title(f"Quantity vs Profit for {product.title()} in {city.title()}")
        plt.xlabel("Quantity")
        plt.ylabel("Profit ($)")
        plt.grid(True)
        plt.show()

    except Exception as e:
        print(f"❗ Error in plot_quantity_vs_profit: {str(e)}")


# ==========================
# STEP 5: User Interaction
# ==========================

print("\n🎯 Welcome to Product Price Spread Predictor!")
user_city = input("Enter city (Atlanta, Chicago, Los Angeles, New York): ").strip()
user_product = input("Enter product name: ").strip()

try:
    user_quantity = int(input("Enter quantity: "))
except ValueError:
    print("❗ Error: Quantity must be an integer.")
    sys.exit(1)

result = predict_best_quantity(user_city, user_product, user_quantity)
print(result)

plot_quantity_vs_profit(user_city, user_product)
