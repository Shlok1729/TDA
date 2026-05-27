# Topological Data Analysis (TDA) Forecasting Dashboard 📈

This project demonstrates how integrating **Topological Data Analysis (TDA)** with traditional machine learning models significantly improves forecasting robustness, especially when dealing with highly chaotic and noisy time-series data.

By utilizing **Persistent Homology**, this engine extracts topological features that capture the underlying "shape" and structural integrity of data. This allows the model to remain robust and detect structural breakdowns (fragility) long before traditional statistical models react, making it a powerful tool for early warning systems across finance, healthcare, and industrial IoT.

## 🌟 Key Features

*   **TDA-Enhanced Machine Learning**: Combines time-delay embedding and persistent homology (via `ripser`) with an XGBoost Regressor to improve prediction accuracy (RMSE/MAE) over traditional baseline models.
*   **Topological Health Monitor**: Calculates a real-time **Structural Fragility Index** using $H_0$ persistence entropy. It acts as an early warning radar for systemic breakdowns.
*   **Domain-Specific Case Studies**:
    *   📉 **Finance & Crypto**: Bitcoin 2021 Flash Crash and the COVID-19 S&P 500 "Black Swan" event. Proves TDA's predictive lead-time advantage.
    *   🫀 **Medical Diagnostics**: ECG rhythm analysis capable of detecting arrhythmic structural loops with extreme noise tolerance.
    *   ⚙️ **Industrial IoT**: Predictive maintenance for turbine bearings, detecting subtle structural wear before mechanical seizure.
*   **Explainable AI (XAI)**: Integrated SHAP value analysis to demonstrate how TDA features (Entropy, Betti Bins) contribute to the model's decision-making process.
*   **Live Signal Simulation**: A real-time processing demo that visualizes the evolution of the signal, its persistence diagram, and the dynamic fragility index.

## 🏗️ Architecture & Technical Stack

*   **UI/Application**: [Streamlit](https://streamlit.io/)
*   **Machine Learning**: [XGBoost](https://xgboost.readthedocs.io/), `scikit-learn`
*   **Topological Data Analysis**: `ripser` (compute persistent homology), `persim` (visualization)
*   **Data Visualization**: `plotly`, `matplotlib`
*   **Data Processing**: `numpy`, `pandas`, `yfinance` (for real-world market data)
*   **Explainability**: `shap`

### Project Structure
```text
TDA/
│
├── app.py                  # Main Streamlit dashboard application
├── requirements.txt        # Python dependencies
└── src/
    ├── dataset.py          # Data generation and fetching (Synthetic, ECG, IoT, Finance)
    ├── models.py           # XGBoost Baseline and TDA-Enhanced Model definitions
    └── tda_features.py     # Core TDA logic: Time-delay embedding, Betti curves, Entropy
```

## 🧠 How the TDA Engine Works

1.  **Time-Delay Embedding**: The 1D time-series signal is projected into a higher-dimensional phase space.
2.  **Persistent Homology**: We compute the persistence of topological features (connected components $H_0$, loops $H_1$) across varying spatial scales (filtrations).
3.  **Feature Extraction**: The raw persistence diagrams are vectorized into scalar features like **Persistence Entropy** and **Betti Curves**.
4.  **Modeling**: These topological features are appended to the raw numerical lags and fed into an XGBoost model.
5.  **Fragility Index**: By monitoring the statistical deviation (Z-score) of the $H_0$ entropy, we can quantify the structural decay of the system in real-time.

## 🚀 Getting Started

### Prerequisites

Make sure you have Python 3.8+ installed. 

### Installation

1.  Clone this repository or navigate to the project directory.
2.  Create a virtual environment (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    ```
3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running the App

Start the Streamlit server by running:
```bash
streamlit run app.py
```

The application will be available in your browser at `http://localhost:8501`.

## 🛡️ License
This project is for demonstration and educational purposes.
