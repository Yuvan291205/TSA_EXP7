# Ex.No: 07                                       AUTO REGRESSIVE MODEL
## NAME YUVAN M
## REG : 212223240188



### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM
```
# 📊 AutoRegressive (AR) Model for Car Price Forecasting (Robust Version)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

# ----------------------------------------------------------
# 1️⃣ Load the dataset
data = pd.read_csv("car.csv")
print("✅ Dataset Loaded Successfully\n")

# Display first few rows and columns
print("Dataset Preview:")
print(data.head())
print("\nColumns:", data.columns.tolist())

# ----------------------------------------------------------
# 2️⃣ Select numeric column for analysis
selected_column = "Price ($)"
print(f"\nUsing column for analysis: {selected_column}")

# ----------------------------------------------------------
# 3️⃣ Clean Price column (convert to numeric)
data[selected_column] = pd.to_numeric(
    data[selected_column].astype(str).str.replace(r'[^0-9.]', '', regex=True),
    errors='coerce'
)
data = data.dropna(subset=[selected_column])  # Drop rows with invalid prices

# ----------------------------------------------------------
# 4️⃣ Convert Date column to datetime and set as index
data["Date"] = pd.to_datetime(data["Date"], errors="coerce")  # Parse dates
data = data.dropna(subset=["Date"])  # Drop rows where date couldn't be parsed
data = data.set_index("Date")
data = data.sort_index()

# Optional: Resample to monthly mean prices
data_monthly = data[selected_column].resample("MS").mean()

# ----------------------------------------------------------
# 5️⃣ Perform Augmented Dickey-Fuller Test for stationarity
result = adfuller(data_monthly.dropna())
print("\nADF Statistic:", result[0])
print("p-value:", result[1])
if result[1] <= 0.05:
    print("✅ Data is stationary.")
else:
    print("⚠️ Data is non-stationary. Differencing may be required.")

# ----------------------------------------------------------
# 6️⃣ Split data into train and test sets (80% train, 20% test)
train_size = int(len(data_monthly) * 0.8)
train_data = data_monthly[:train_size]
test_data = data_monthly[train_size:]

# ----------------------------------------------------------
# 7️⃣ Fit AutoRegressive (AR) model
lag_order = 5  # Can adjust based on PACF
model = AutoReg(train_data, lags=lag_order)
model_fit = model.fit()

print(f"\n✅ Model trained with lag order = {lag_order}")
print(model_fit.summary())

# ----------------------------------------------------------
# 8️⃣ Plot ACF and PACF safely for small datasets
max_lags = min(20, len(data_monthly.dropna()) // 2)  # Max lags allowed

plt.figure(figsize=(10, 5))
plot_acf(data_monthly.dropna(), lags=max_lags, alpha=0.05)
plt.title("Autocorrelation Function (ACF)")
plt.show()

plt.figure(figsize=(10, 5))
plot_pacf(data_monthly.dropna(), lags=max_lags, alpha=0.05)
plt.title("Partial Autocorrelation Function (PACF)")
plt.show()

# ----------------------------------------------------------
# 9️⃣ Make predictions using the AR model
predictions = model_fit.predict(
    start=len(train_data),
    end=len(train_data) + len(test_data) - 1
)

# ----------------------------------------------------------
# 🔟 Evaluate model performance
mse = mean_squared_error(test_data, predictions)
rmse = np.sqrt(mse)
print(f"\n📈 Model Performance:")
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)

# ----------------------------------------------------------
# 1️⃣1️⃣ Plot train data, test data, and predictions
plt.figure(figsize=(12, 6))
plt.plot(train_data.index, train_data, label="Train Data")
plt.plot(test_data.index, test_data, label="Test Data", color="orange")
plt.plot(test_data.index, predictions, label="Predicted", linestyle="--", color="red")
plt.title("AR Model Predictions vs Test Data (Car Prices)")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend()
plt.grid(True)
plt.show()
```
### OUTPUT:

Dataset Preview:
         Car_id      Date Customer Name Gender Annual Income  \
0  C_CND_000001  1/2/2022     Geraldine   Male         13500   
1  C_CND_000002  1/2/2022           Gia   Male       1480000   
2  C_CND_000003  1/2/2022        Gianna   Male       1035000   
3  C_CND_000004  1/2/2022       Giselle   Male         13500   
4  C_CND_000005  1/2/2022         Grace   Male       1465000   

                           Dealer_Name   Company       Model  \
0  Buddy Storbeck's Diesel Service Inc      Ford  Expedition   
1                     C & M Motors Inc     Dodge     Durango   
2                          Capitol KIA  Cadillac    Eldorado   
3               Chrysler of Tri-Cities    Toyota      Celica   
4                    Chrysler Plymouth     Acura          TL   

                      Engine Transmission       Color Price ($)  Dealer_No   \
0  DoubleÂ Overhead Camshaft         Auto       Black     26000  06457-3834   
1  DoubleÂ Overhead Camshaft         Auto       Black     19000  60504-7114   
2          Overhead Camshaft       Manual         Red     31500  38701-8047   
3          Overhead Camshaft       Manual  Pale White     14000  99301-3882   
4  DoubleÂ Overhead Camshaft         Auto         Red     24500  53546-9427   

  Body Style      Phone Dealer_Region  
0        SUV  8264678.0    Middletown  
1        SUV  6848189.0        Aurora  
2  Passenger  7298798.0    Greenville  
3        SUV  6257557.0         Pasco  
4  Hatchback  7081483.0    Janesville  

Columns: ['Car_id', 'Date', 'Customer Name', 'Gender', 'Annual Income', 'Dealer_Name', 'Company', 'Model', 'Engine', 'Transmission', 'Color', 'Price ($)', 'Dealer_No ', 'Body Style', 'Phone', 'Dealer_Region']

Using column for analysis: Price ($)

ADF Statistic: -4.586386321877345
p-value: 0.00013684203144454668
✅ Data is stationary.

✅ Model trained with lag order = 5
                            AutoReg Model Results                             
==============================================================================
Dep. Variable:              Price ($)   No. Observations:                   19
Model:                     AutoReg(5)   Log Likelihood                -106.035
Method:               Conditional MLE   S.D. of innovations            471.047
Date:                Mon, 06 Oct 2025   AIC                            226.069
Time:                        10:20:21   BIC                            230.543
Sample:                    06-01-2022   HQIC                           225.655
                         - 07-01-2023                                         
================================================================================
                   coef    std err          z      P>|z|      [0.025      0.975]
--------------------------------------------------------------------------------
const         4.948e+04   1.15e+04      4.292      0.000    2.69e+04    7.21e+04
Price ($).L1    -0.2499      0.229     -1.092      0.275      -0.698       0.198
Price ($).L2     0.2586      0.196      1.318      0.188      -0.126       0.643
Price ($).L3     0.2234      0.239      0.936      0.349      -0.244       0.691
Price ($).L4    -0.1501      0.192     -0.782      0.434      -0.526       0.226
Price ($).L5    -0.8410      0.264     -3.183      0.001      -1.359      -0.323
                                    Roots                                    
=============================================================================
                  Real          Imaginary           Modulus         Frequency
-----------------------------------------------------------------------------
AR.1           -1.0064           -0.0000j            1.0064           -0.5000
AR.2           -0.4597           -0.9315j            1.0388           -0.3230
AR.3           -0.4597           +0.9315j            1.0388            0.3230
AR.4            0.8736           -0.5759j            1.0464           -0.0928
AR.5            0.8736           +0.5759j            1.0464            0.0928
-----------------------------------------------------------------------------
<Figure size 1000x500 with 0 Axes>
<img width="568" height="435" alt="download" src="https://github.com/user-attachments/assets/6801fdc5-62b5-4627-9f6a-a8e8be76e5fa" />
<img width="568" height="435" alt="download" src="https://github.com/user-attachments/assets/1f1d8439-c836-43c1-90da-77129c0988ca" />

📈 Model Performance:
Mean Squared Error (MSE): 498105.26395089633
Root Mean Squared Error (RMSE): 705.7657288016303

<img width="1047" height="547" alt="download" src="https://github.com/user-attachments/assets/d38f0696-e45e-455a-8881-ca1f384e877f" />


### RESULT:
Thus we have successfully implemented the auto regression function using python.
