# 🤖 KNN Optimization and Performance Analysis App

A web-based machine learning experiment built with **Streamlit** to visualize and optimize **K-Nearest Neighbors (KNN)** performance on the Car Evaluation dataset.

## 🚀 Features
- Upload any dataset or use the built-in Car Evaluation dataset.
- Choose K values, metrics, and training proportions.
- Automatic **GridSearchCV** hyperparameter tuning.
- Real-time **accuracy plots**, **best parameter summaries**, and **prediction time benchmarks**.

## 🧠 Tech Stack
- **Python**, **scikit-learn**, **pandas**, **numpy**
- **Streamlit** for web interface
- **matplotlib**, **seaborn** for visualization

## 📊 How to Run Locally
```bash
pip install streamlit scikit-learn pandas seaborn matplotlib numpy
streamlit run knn_app.py
