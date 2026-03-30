# 🚀 Hybrid Fuzzy-Neural Network for Wind Power Forecasting

##  Project Overview

This project presents a hybrid forecasting system that integrates **Fuzzy Time Series (FTS)** with **Neural Networks (MLP)** to predict wind turbine power output using multivariate sensor data.

The model is designed to handle **non-stationary time series**, where traditional methods fail due to noise, uncertainty, and rapidly changing patterns.

This project focuses on real-world smart grid applications, enabling accurate and interpretable wind power forecasting for renewable energy systems.

---

##  Problem Statement

Accurate prediction of wind power is challenging due to:

* Nonlinear relationships between environmental variables
* High variability in wind conditions
* Noisy and uncertain sensor data

Traditional models such as ARIMA and linear regression fail to capture these complexities.

---

##  Proposed Solution

This project introduces a **hybrid model** that combines:

* 🔹 **Fuzzy Time Series (FTS)**
  Handles uncertainty and provides interpretable rules

* 🔹 **Neural Network (MLP)**
  Learns nonlinear relationships between multiple features

* 🔹 **Hybrid Model (FTS + NN)**
  Combines both approaches for improved accuracy and stability

---

##  Features Used

The model uses multivariate sensor inputs:

* Wind speed
* Wind direction
* Air density
* Temperature
* Pressure
* Humidity
* Rotor speed
* Blade pitch
* Turbine yaw
* Vibration

**Target variable:**

* Power output

---

##  Dataset

This project uses a synthetic wind energy dataset generated to simulate real-world turbine sensor behavior under varying environmental conditions. The dataset captures nonlinear and non-stationary patterns required for evaluating hybrid forecasting models.

* Dataset file: `data/sample_wind_data.csv`
* Contains multivariate sensor readings and corresponding power output
* Designed for experimentation and model validation

**Note:** The dataset is a representative sample. Future work includes validation using real-world SCADA datasets.

---

##  System Architecture

The system consists of:

1. Data preprocessing (cleaning, scaling, normalization)
2. Fuzzy interval generation and fuzzification
3. Neural network training (MLP)
4. Hybrid prediction combining FTS and NN
5. Visualization through GUI and dashboard

---

##  Results

The hybrid model demonstrates:

* Lower prediction error (MAE, RMSE, MAPE)
* Improved stability under fluctuating wind conditions
* Better performance compared to traditional models

---

##  Application Interface

The system includes:

* Tkinter-based GUI
* Plotly/Dash dashboard
* Real-time prediction visualization

---

##  How to Run the Project

### Step 1: Install dependencies

pip install -r requirements.txt

### Step 2: Run the application

python main.py

---

##  Project Structure

fuzzy-neural-wind-forecast/
│── main.py
│── README.md
│── requirements.txt
│
├── data/
│   ├── sample_wind_data.csv
│
├── documents/
│   ├── report.pdf
│   ├── poster.pdf
│   ├── presentation.pdf
│
├── assets/
│   ├── dashboard.png

---

##  Project Documents

This repository includes:

* Full project report
* Research poster
* Presentation slides

---

##  Demo

Demo video will be added soon.

---

##  Key Contributions

* Hybrid AI model for non-stationary time series
* Integration of fuzzy reasoning and deep learning
* End-to-end system with GUI and visualization
* Real-world application in renewable energy

---

##  Author

Digumurthy Sruthi Sarika

---

##  Note

This project was developed as part of a capstone in Computer Science and Engineering, focusing on intelligent forecasting systems for real-world smart energy and renewable grid applications.

