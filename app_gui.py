import tkinter as tk
from tkinter import messagebox, ttk
from mainfile import predict_best_quantity, plot_quantity_vs_profit
import os

# ===== Sample Cities & Products (You can add more) =====
CITIES = ["Bangalore", "Chennai", "Mumbai", "Delhi", "Hyderabad"]
PRODUCTS = ["Wheat", "Rice", "Maize", "Sugarcane", "Cotton"]

# ================ FUNCTIONALITIES ================

def predict():
    city = city_combo.get()
    product = product_combo.get()
    quantity = quantity_entry.get()

    if not city or not product or not quantity:
        messagebox.showerror("Input Error", "Please fill all fields!")
        return

    try:
        quantity = int(quantity)
    except ValueError:
        messagebox.showerror("Input Error", "Quantity must be an integer!")
        return

    result = predict_best_quantity(city, product, quantity)
    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, result)

    # Enable View Report button
    report_button.config(state=tk.NORMAL)


def open_report():
    city = city_combo.get().lower()
    product = product_combo.get().lower()

    filename = f"prediction_report_{product}_{city}.txt"

    if os.path.exists(filename):
        os.startfile(filename)
    else:
        messagebox.showerror("Report Not Found", "Report file not found!")


def show_visualization():
    city = city_combo.get()
    product = product_combo.get()

    if not city or not product:
        messagebox.showerror("Input Error", "Select city & product first!")
        return

    plot_quantity_vs_profit(city, product)


def clear_fields():
    city_combo.set('')
    product_combo.set('')
    quantity_entry.delete(0, tk.END)
    result_text.delete(1.0, tk.END)
    report_button.config(state=tk.DISABLED)

# ================ GUI DESIGN ================

root = tk.Tk()
root.title("🌾 Product Price Spread Predictor")
root.geometry("600x530")
root.configure(bg="#f0f0f0")

title_label = tk.Label(root, text="🌟 Price Spread Prediction", font=("Arial", 18, "bold"), bg="#f0f0f0", fg="green")
title_label.pack(pady=10)

frame = tk.Frame(root, bg="#f0f0f0")
frame.pack(pady=10)

# City Dropdown
tk.Label(frame, text="City:", font=("Arial", 12), bg="#f0f0f0").grid(row=0, column=0, padx=5, pady=5, sticky="e")
city_combo = ttk.Combobox(frame, values=CITIES, font=("Arial", 12), state="readonly")
city_combo.grid(row=0, column=1, padx=5, pady=5)

# Product Dropdown
tk.Label(frame, text="Product:", font=("Arial", 12), bg="#f0f0f0").grid(row=1, column=0, padx=5, pady=5, sticky="e")
product_combo = ttk.Combobox(frame, values=PRODUCTS, font=("Arial", 12), state="readonly")
product_combo.grid(row=1, column=1, padx=5, pady=5)

# Quantity Entry
tk.Label(frame, text="Quantity:", font=("Arial", 12), bg="#f0f0f0").grid(row=2, column=0, padx=5, pady=5, sticky="e")
quantity_entry = tk.Entry(frame, font=("Arial", 12))
quantity_entry.grid(row=2, column=1, padx=5, pady=5)

# Buttons
button_frame = tk.Frame(root, bg="#f0f0f0")
button_frame.pack(pady=10)

predict_button = tk.Button(button_frame, text="🚀 Predict", font=("Arial", 12, "bold"), bg="green", fg="white", command=predict)
predict_button.grid(row=0, column=0, padx=5)

report_button = tk.Button(button_frame, text="📄 View Report", font=("Arial", 12), bg="#007acc", fg="white", state=tk.DISABLED, command=open_report)
report_button.grid(row=0, column=1, padx=5)

viz_button = tk.Button(button_frame, text="📊 Show Visualization", font=("Arial", 12), bg="#6a0dad", fg="white", command=show_visualization)
viz_button.grid(row=0, column=2, padx=5)

clear_button = tk.Button(button_frame, text="🧹 Clear", font=("Arial", 12), bg="#e67300", fg="white", command=clear_fields)
clear_button.grid(row=0, column=3, padx=5)

# Result Display
result_text = tk.Text(root, height=10, width=65, font=("Arial", 10))
result_text.pack(pady=10)

root.mainloop()
