# 🚀  Hybrid Fuzzy-Neural Network for Wind Power Forecasting

---

## Abstract

Accurate wind power forecasting is essential for efficient energy management and grid stability. Traditional statistical methods often fail to capture the non-linear and non-stationary nature of wind data. This project proposes a hybrid forecasting framework that combines Fuzzy Time Series (FTS) and Neural Networks to improve prediction accuracy. By integrating rule-based reasoning with data-driven learning, the system provides a more robust and adaptive approach to time-series forecasting.

---

## Problem Statement

Wind energy generation is highly volatile due to environmental fluctuations. Predicting wind power output accurately is challenging because of:

* Non-linearity in data patterns
* Temporal dependencies
* Noise and uncertainty

This project aims to design a hybrid system that can effectively model these complexities and improve forecasting performance.

---

## Methodology

The system integrates three approaches:

### 1. Fuzzy Time Series (FTS)

* Handles uncertainty using fuzzy logic
* Converts numerical data into linguistic intervals
* Captures approximate trends

### 2. Neural Network (Multivariate)

* Learns complex non-linear relationships
* Uses historical data to model dependencies
* Improves prediction accuracy over traditional methods

### 3. Hybrid Model (FTS + Neural Network)

* Combines fuzzy rule-based reasoning with neural learning
* Enhances generalization capability
* Reduces forecasting error

---

## System Architecture

The workflow of the system is as follows:

1. Data Loading
2. Preprocessing and normalization
3. Model training (FTS, Neural Network, Hybrid)
4. Prediction generation
5. Performance evaluation
6. Visualization of results

---

## Dataset

The dataset used in this project consists of wind energy observations including:

* Wind speed
* Power output
* Time-based features

The dataset is structured to simulate real-world wind forecasting scenarios and is used for training and evaluating all models.

---

## Evaluation Metrics

The performance of the models is evaluated using:

* **MAE (Mean Absolute Error)**
* **RMSE (Root Mean Squared Error)**
* **MAPE (Mean Absolute Percentage Error)**

Lower values indicate better model performance.

---

## Results and Analysis

The experimental results demonstrate clear differences in model performance:

* The **FTS model** shows higher error due to its limited ability to model complex patterns
* The **Neural Network model** performs significantly better by capturing non-linear relationships
* The **Hybrid model** achieves a balance between interpretability and accuracy

The hybrid approach reduces prediction errors compared to traditional methods, making it more suitable for real-world forecasting applications.

---

## Visual Results

### Model Performance Dashboard

![Dashboard](assets/dashboard.png)

### Model Comparison

![Model Comparison](assets/model_comparison.png)

### Error Distribution

![Error Distribution](assets/error_distribution.png)

### Forecast Results

![Forecast](assets/forecast_full.png)

### Detailed Forecast (Last 50 Samples)

![Forecast Zoom](assets/forecast_last50.png)

---

## Key Contributions

* Development of a hybrid fuzzy-neural forecasting model
* Integration of rule-based and machine learning approaches
* Comparative evaluation of multiple models
* Visualization-driven analysis of forecasting performance
* Implementation of a GUI-based system

---

## Limitations

* The dataset size is moderate and may not fully represent large-scale real-world scenarios
* The neural network architecture is relatively simple
* External environmental factors are not fully incorporated

---

## Future Work

* Incorporating deep learning models such as LSTM for time-series forecasting
* Using large-scale real-world datasets for improved generalization
* Enhancing feature engineering with additional environmental variables
* Deploying the system as a real-time forecasting application

---

## Demo Video

[Watch Demo Video](https://drive.google.com/file/d/1n084Lqtw0tJVry5EKYe0QBKr7juEYWPX/view?usp=sharing)

---

## Conclusion

This project demonstrates that combining fuzzy logic with neural networks significantly improves wind power forecasting accuracy. The hybrid model effectively captures both uncertainty and complex patterns in the data, providing a reliable solution for time-series prediction tasks.

---

## Author

Digumurthy Sruthi Sarika
