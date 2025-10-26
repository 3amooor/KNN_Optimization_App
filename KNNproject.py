import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="KNN Optimization App", layout="wide")

st.title("🤖 KNN Optimization and Performance Analysis")
st.write("Experiment with different training portions and K values using the **Car Evaluation dataset** or your own CSV file.")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
else:
    st.info("Using default dataset: `CAR_EVALUATION.csv`")
    df = pd.read_csv("38fc19ce-488e-419c-a410-a0b9540935a3.csv") 

st.subheader("Dataset Preview")
st.dataframe(df.head())

if "Target" not in df.columns:
    st.error(" Your dataset must contain a column named 'Target'.")
    st.stop()

x = df.drop("Target", axis=1)
y = df["Target"]

x_temp, x_test, y_temp, y_test = train_test_split(x, y, test_size=428/1728, random_state=42, shuffle=True)
x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=300/1300, random_state=42, shuffle=True)

all_features = pd.concat([x_train, x_val, x_test])
for col in all_features.columns:
    if all_features[col].dtype == 'object':
        le = LabelEncoder()
        all_features[col] = le.fit_transform(all_features[col])

x_train = all_features.iloc[:len(x_train)]
x_val = all_features.iloc[len(x_train):len(x_train)+len(x_val)]
x_test = all_features.iloc[len(x_train)+len(x_val):]

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_val_scaled = scaler.transform(x_val)
x_test_scaled = scaler.transform(x_test)

st.sidebar.header("Experiment Settings")
neighbors_list = st.sidebar.multiselect("K values for Grid Search", [3,5,7,9,11], default=[3,5,7,9,11])
metrics_list = st.sidebar.multiselect("Distance Metrics", ["euclidean","manhattan"], default=["euclidean","manhattan"])
train_portions = np.arange(0.1, 1.1, 0.1)
run_button = st.sidebar.button("Run KNN Experiment")

if run_button:
    val_accuracies, test_accuracies, best_params_list = [], [], []
    param_grid = {
        'n_neighbors': neighbors_list,
        'weights': ['uniform', 'distance'],
        'metric': metrics_list
    }

    progress = st.progress(0)
    for i, portion in enumerate(train_portions):
        portion_size = int(portion * len(x_train_scaled))
        x_train_portion = x_train_scaled[:portion_size]
        y_train_portion = y_train.iloc[:portion_size]

        knn = KNeighborsClassifier()
        grid_search = GridSearchCV(knn, param_grid, cv=5, n_jobs=-1)
        grid_search.fit(x_train_portion, y_train_portion)
        best_knn = grid_search.best_estimator_

        y_val_pred = best_knn.predict(x_val_scaled)
        y_test_pred = best_knn.predict(x_test_scaled)

        val_accuracies.append(accuracy_score(y_val, y_val_pred))
        test_accuracies.append(accuracy_score(y_test, y_test_pred))
        best_params_list.append(grid_search.best_params_)
        progress.progress((i+1)/len(train_portions))

    st.subheader("Validation vs Testing Accuracy")
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(train_portions*100, val_accuracies, marker='o', label='Validation Accuracy')
    ax.plot(train_portions*100, test_accuracies, marker='s', label='Testing Accuracy')
    ax.set_title("KNN Performance Across Training Portions")
    ax.set_xlabel("Training Portion (%)")
    ax.set_ylabel("Accuracy")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    results_df = pd.DataFrame({
        'Training Portion (%)': train_portions*100,
        'Validation Accuracy': val_accuracies,
        'Testing Accuracy': test_accuracies,
        'Best Params': best_params_list
    })
    st.dataframe(results_df)

    st.subheader("K Value Optimization")
    k_values = range(1, 11)
    val_accuracies_k = []

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_train_scaled, y_train)
        y_val_pred = knn.predict(x_val_scaled)
        val_accuracies_k.append(accuracy_score(y_val, y_val_pred))

    best_k = k_values[np.argmax(val_accuracies_k)]
    best_accuracy = max(val_accuracies_k)
    st.write(f"**Best K:** {best_k} | **Validation Accuracy:** {best_accuracy:.4f}")

    fig2, ax2 = plt.subplots(figsize=(8,5))
    ax2.plot(k_values, val_accuracies_k, marker='o', color='teal')
    ax2.set_title("Validation Accuracy vs K Value")
    ax2.set_xlabel("K")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True)
    st.pyplot(fig2)

    st.subheader("⏱️ Prediction Time Comparison")
    cases = [(0.1, 2), (1.0, 2), (0.1, 10), (1.0, 10)]
    labels, predict_times = [], []

    for portion, k in cases:
        size = int(len(x_train_scaled)*portion)
        x_subset = x_train_scaled[:size]
        y_subset = y_train.iloc[:size]
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_subset, y_subset)

        start = time.time()
        knn.predict(x_test_scaled)
        end = time.time()
        predict_times.append(end - start)
        labels.append(f"{int(portion*100)}%_K={k}")

    fig3, ax3 = plt.subplots()
    ax3.bar(labels, predict_times, color='skyblue')
    ax3.set_title("Prediction Time Comparison")
    ax3.set_xlabel("Case")
    ax3.set_ylabel("Prediction Time (s)")
    ax3.grid(axis='y', linestyle='--', alpha=0.6)
    st.pyplot(fig3)

st.markdown("---")
