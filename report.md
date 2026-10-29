# Capstone Project 14: AI-Based Bluetooth Signal Quality Analysis

## Abstract
This project demonstrates a machine-learning workflow for estimating Bluetooth link quality from signal and connection indicators. A Random Forest classifier maps RSSI, packet loss, latency, distance, interference level, obstacle count, and connection status to one of three classes: good, moderate, or poor. A Streamlit interface allows a user to enter conditions and view a prediction. The included dataset is synthetic and is intended for software demonstration only; it is not a record of measurements from Bluetooth hardware.

## 1. Introduction
Bluetooth is a short-range wireless technology used by peripherals, sensors, wearables, and embedded devices. Link quality can change with distance, obstacles, radio interference, antenna orientation, and device implementation. A model that combines multiple indicators may provide a convenient estimate of link condition.

## 2. Problem statement
Build a prototype that predicts Bluetooth link quality and investigates how distance and interference relate to the predicted class.

## 3. Objectives
1. Prepare a structured dataset of Bluetooth-like link measurements.
2. Preprocess numerical and categorical features.
3. Train a supervised classification model.
4. Evaluate the model using accuracy, macro F1-score, and a confusion matrix.
5. Display predictions and basic data summaries in a dashboard.

## 4. System requirements
- Python 3.10+
- pandas, NumPy, scikit-learn, Matplotlib, joblib, Streamlit
- Laptop or desktop computer
- Optional real Bluetooth device and measurement utilities for field data collection

## 5. Methodology
### 5.1 Input features
RSSI (dBm), packet loss (%), latency (ms), distance (m), interference category, obstacle count, and connection status.

### 5.2 Dataset
The repository includes 1,800 reproducibly generated synthetic rows. The generator simulates plausible trends, including weaker RSSI and higher latency/packet loss as distance, interference, and obstacles increase. Labels are derived from a rule-based score for demonstration. Consequently, the model's results measure how well it learns these synthetic patterns, not real Bluetooth performance.

### 5.3 Preprocessing
Missing rows for required fields are removed. Numerical columns are standardized and categorical columns are one-hot encoded using a scikit-learn `ColumnTransformer`.

### 5.4 Model
A Random Forest classifier with 250 trees is trained. An 80/20 stratified train-test split is used with a fixed random seed for reproducibility. The entire preprocessing-plus-classifier pipeline is saved using joblib.

### 5.5 Evaluation
The training script reports accuracy, macro F1-score, and per-class precision/recall/F1. It also creates a confusion matrix and class-distribution plot in `outputs/`. Run `python train_model.py` to generate exact metrics for the environment and dataset.

## 6. System architecture
**Inputs → Data cleaning → Feature preprocessing → Random Forest classifier → Quality class + probabilities → Streamlit dashboard**

## 7. Running the project
```bash
pip install -r requirements.txt
python generate_dataset.py
python train_model.py
streamlit run app.py
```

## 8. Results and discussion
The generated metrics are written to `outputs/metrics.json`. Because the dataset and its labels are synthetic, scores can appear high and should not be presented as evidence of accuracy on physical Bluetooth links. In a final academic submission, insert results from actual repeated measurements and discuss variation by distance and interference.

## 9. Real-world data collection plan
1. Choose a transmitter/receiver pair and record device model, OS, Bluetooth version, and environment.
2. Measure RSSI at fixed distance intervals (for example, 1, 3, 5, 10, and 15 m), with repeated trials.
3. Estimate packet loss using a known number of transmitted packets and count successful receptions.
4. Measure latency with a repeatable request/response test.
5. Repeat in low-, medium-, and high-interference environments and document obstacles.
6. Define quality labels before training, avoiding labels that leak the answer directly from input fields.
7. Split by session or environment where possible so near-duplicate samples do not appear in both training and testing.
8. Retrain and report results from a held-out real measurement set.

## 10. Limitations
- Synthetic data is not a substitute for physical measurements.
- Bluetooth implementations differ across chipsets, antennas, versions, and operating systems.
- Results can be affected by measurement noise and class-label definitions.
- Prediction probabilities are model estimates, not guaranteed confidence.

## 11. Conclusion
The project provides an end-to-end educational prototype for preparing data, training a classifier, evaluating it, and demonstrating predictions through a dashboard. The next step is collecting and validating real Bluetooth measurements.

## References
1. scikit-learn documentation, https://scikit-learn.org/
2. pandas documentation, https://pandas.pydata.org/docs/
3. Streamlit documentation, https://docs.streamlit.io/
4. Bluetooth SIG, Bluetooth technology overview, https://www.bluetooth.com/learn-about-bluetooth/
