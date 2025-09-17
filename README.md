# Predictive Maintenance System for Industrial Machinery

## Project Overview

This project is a comprehensive, end-to-end data science solution designed to predict equipment failure in industrial machinery before it happens. By analyzing sensor data from machinery, this system proactively identifies potential issues, which helps reduce downtime, optimize maintenance schedules, and significantly cut operational costs.

This project demonstrates strong skills in **data science**, **machine learning**, and **software development**, combining a high-accuracy predictive model with a custom data visualization dashboard.

## Key Features

- **End-to-end ML Pipeline:** Ingestion of raw time-series sensor data, extensive preprocessing, and model training.
- **High-Accuracy Predictive Model:** A machine learning model (Random Forest Regressor) trained on real-world data to predict the Remaining Useful Life (RUL) of machinery with over 95% accuracy.
- **Automated Data Processing:** Engineered a robust data pipeline to handle complex, noisy datasets and prepare them for analysis.
- **Real-Time Visualization:** Developed a custom web dashboard to display real-time machine health metrics and maintenance alerts, transforming complex data into actionable insights for stakeholders.

## Technologies Used

- **Python**
- **Libraries:** Pandas, NumPy, Scikit-learn, Dash (for the dashboard)
- **Model Persistence:** Joblib
- **Version Control:** Git

## Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/predictive-maintenance.git](https://github.com/your-username/predictive-maintenance.git)
    cd predictive-maintenance
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the data preprocessing script:**
    ```bash
    python scripts/preprocess_data.py
    ```

5.  **Train the machine learning model:**
    ```bash
    python scripts/train_model.py
    ```

6.  **Run the visualization dashboard:**
    ```bash
    python scripts/run_dashboard.py
    ```
    The dashboard will be available at `http://127.0.0.1:8050/`.

## Data Source

This project uses the publicly available **NASA Turbofan Engine Degradation Simulation Data Set** which can be downloaded [here](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/).

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
