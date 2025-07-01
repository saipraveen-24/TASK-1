# 📊 Data Pipeline Development – Task 1 (CodTech Internship)

## 🧑‍💻 Project Description

This project implements a **complete ETL (Extract, Transform, Load) pipeline** using Python, `pandas`, and `scikit-learn`. It processes a raw dataset (`titanic.csv`), performs preprocessing and transformation, and outputs clean, ready-to-use machine learning data files.

---

## 📂 Project Structure

.
├── etl_pipeline.py # Main ETL script
├── data/
│ └── titanic.csv # Raw input dataset
├── processed_data/
│ ├── features.csv # Preprocessed features (scaled + encoded)
│ └── target.csv # Target labels (e.g., Survived)
└── README.md # Project documentation

---

## 🛠️ Tools & Libraries Used

- Python 3.x
- pandas
- scikit-learn
- NumPy

---

## 🔁 ETL Process Overview

1. **Extract**
   - Reads the raw Titanic dataset from `data/titanic.csv`.

2. **Transform**
   - Handles missing values.
   - Encodes categorical variables using `OneHotEncoder`.
   - Scales numerical features using `StandardScaler`.

3. **Load**
   - Saves processed features to `processed_data/features.csv`.
   - Saves target labels to `processed_data/target.csv`.

---

## ▶️ How to Run

1. **Install dependencies:**

```bash
pip install pandas scikit-learn numpy
