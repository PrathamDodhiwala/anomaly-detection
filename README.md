# 🔍 Financial Transaction Anomaly Detection App (Python + Streamlit)

A complete end-to-end **Anomaly Detection Web App** for financial transactions built using **Python** and **Streamlit**. This project detects suspicious or unusual transactions using unsupervised machine learning techniques and provides an interactive UI for analysis and visualization.


## 🚀 Features

* 📂 Upload your own transaction dataset or use built-in synthetic data
* 🧠 Multiple Anomaly Detection Models:

  * Isolation Forest
  * Local Outlier Factor (LOF)
  * Autoencoder (optional – requires TensorFlow)
* 🔧 Automatic Feature Engineering:

  * Time-based features (hour, day, month)
  * Scaling of numerical data
  * Encoding of categorical variables
* 📊 Interactive Visualizations:

  * PCA scatter plot (normal vs anomalies)
  * Time-series anomaly detection plot
  * Anomaly score distribution
* 🔎 Inspect individual anomalous transactions
* 📥 Download flagged anomalies as CSV
* ⚡ Fast, interactive UI with Streamlit


## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib, Seaborn
* Streamlit
* TensorFlow (optional for Autoencoder)


## 📂 Project Structure

```
fraud_anomaly_detection_app/
│
├── fraud_anomaly_app.py    # Main Streamlit application
├── requirements.txt       # Dependencies (optional)
└── README.md              # Project documentation
```


## ⚙️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/fraud-anomaly-detection-app.git
cd fraud-anomaly-detection-app
```

### 2️⃣ Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit
```

### (Optional for Autoencoder)

```bash
pip install tensorflow
```


## ▶️ Running the App

```bash
streamlit run fraud_anomaly_app.py
```

Then open:

```
http://localhost:8501
```


## 📊 Dataset Format

Your CSV should ideally include:

```
transaction_id,timestamp,amount,merchant,category,user_id
tx001,2024-01-01 10:30:00,120.50,Amazon,shopping,101
tx002,2024-01-01 11:00:00,15.20,Uber,transport,102
```

### Recommended Columns:

* `timestamp` → transaction time
* `amount` → transaction value
* `merchant` → merchant name
* `category` → transaction type
* `user_id` → customer identifier


## ⚡ How It Works

1. Load dataset (upload or synthetic sample)
2. Select:

   * Datetime column
   * Numeric & categorical features
3. Choose anomaly detection model
4. Set contamination level (expected anomaly %)
5. Run detection
6. View:

   * Flagged anomalies
   * Visualizations
   * Download results


## 🧠 Models Used

### Isolation Forest

* Detects anomalies by isolating observations
* Fast and efficient
* Works well for high-dimensional data

### Local Outlier Factor (LOF)

* Detects anomalies based on local density deviation
* Good for clustered datasets

### Autoencoder (Optional)

* Neural network reconstruction-based anomaly detection
* High accuracy for complex patterns


## 📈 Feature Engineering

Automatically generates:

* Time-based features:

  * Hour of day
  * Day of week
  * Month
* Encoded categorical variables
* Scaled numerical features


## 📉 Output

* `anomaly_label` → 1 (anomaly), 0 (normal)
* `anomaly_score` → higher means more suspicious


## 🎯 Use Cases

* Fraud detection
* Banking transaction monitoring
* Payment anomaly detection
* Cybersecurity anomaly detection
* Financial risk analysis


## ⚠️ Notes

* This is an **unsupervised model** → anomalies require human verification
* Best results with larger datasets
* Tune contamination carefully (0.01–0.05 recommended)


## 📌 Future Improvements

* SHAP explainability
* Real-time streaming detection
* Per-user anomaly models
* Deep learning enhancements
* Deployment on cloud


## 🙌 Contributing

Contributions are welcome! Feel free to fork and improve.

## 📜 License

This project is open-source under the MIT License.

## Author

Pratham Dodhiwala

## ⭐ Support

If you found this useful, give it a ⭐ on GitHub!

