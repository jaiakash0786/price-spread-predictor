# price-spread-predictor
Developed a Streamlit web app to predict product price spreads and maximize profits for shop owners and farmers. Used Python, Pandas, NumPy, Matplotlib, scikit-learn (RandomForestRegressor) for data processing, modeling, and visualization, with downloadable PPTX reports including prediction graphs.
# Product Price Spread Predictor

## Overview
This is a Streamlit web application designed to predict product price spreads and help shop owners and farmers maximize profits. The app predicts optimal purchase quantities, expected profits, and the best city for export using historical price data and machine learning.  

## Features
- Predict price spread and optimal purchase quantity for shop owners.
- Recommend the best city for farmers to export products to maximize profit.
- Display inputs, predictions, and visualizations on the web interface.
- Generate downloadable PPTX reports including detailed graphs.
- Uses Random Forest Regressor for spread prediction with model evaluation metrics.

## Technologies Used
- **Python** for scripting and logic
- **Streamlit** for interactive web interface
- **Pandas & NumPy** for data manipulation
- **Matplotlib** for visualizations
- **scikit-learn (RandomForestRegressor)** for prediction modeling
- **python-pptx** for generating downloadable reports

## price-spread-predictor/
│
├── stapp.py               # Main Streamlit app
├── product_prices.csv     # Dataset file
├── rf_price_predictor.pkl # Trained Random Forest model
├── README.md              # Project documentation
└── requirements.txt       # Python dependencies


## Dataset
- CSV file: `product_prices.csv` containing product names, farm prices, retail prices across multiple cities, and average spreads.

## How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/jaiakash0786/price-spread-predictor
   cd price-spread-predictor


2.Install dependencies:

pip install -r requirements.txt


3.Run the Streamlit app:

streamlit run stapp.py


4.Open the URL shown in the terminal to access the app.

requirements.txt:
streamlit
pandas
numpy
matplotlib
scikit-learn
joblib
python-pptx

output:
<img width="1528" height="448" alt="image" src="https://github.com/user-attachments/assets/9ab4a5f5-8458-411a-8d8f-7cf1e9eb5b57" />
<img width="1809" height="759" alt="image" src="https://github.com/user-attachments/assets/c29b92f8-1cf7-4dcf-b4c3-1ef12d390cb9" />
<img width="1108" height="443" alt="image" src="https://github.com/user-attachments/assets/14a960aa-5132-499c-9c1d-4d6bd30cf446" />

<img width="1735" height="594" alt="image" src="https://github.com/user-attachments/assets/0a018bb8-168f-4a78-8cc2-90df2f727c2c" />
<img width="1271" height="830" alt="image" src="https://github.com/user-attachments/assets/90a61d79-43de-4a9e-8276-9833ca8a1ff7" />
(check for ppt in exportedtexts folder or your downloads)

