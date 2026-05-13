import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ---------------------------------------------------
# PAGE TITLE
# ---------------------------------------------------

st.title("🌳 Decision Tree Regression App")
st.write("Regression using Decision Tree Regressor")

# ---------------------------------------------------
# CREATE DATASET
# ---------------------------------------------------

X, y = make_regression(
    n_samples=200,
    n_features=1,
    noise=15,
    random_state=42
)

df = pd.DataFrame(X, columns=["Feature"])
df["Target"] = y

# ---------------------------------------------------
# SIDEBAR - HYPERPARAMETERS
# ---------------------------------------------------

st.sidebar.header("Hyperparameters")

max_depth = st.sidebar.slider(
    "Max Depth",
    min_value=1,
    max_value=20,
    value=5
)

splitter = st.sidebar.selectbox(
    "Select Splitter",
    ["best", "random"]
)

test_size = st.sidebar.slider(
    "Test Size",
    min_value=0.1,
    max_value=0.5,
    value=0.2
)

# ---------------------------------------------------
# TRAIN TEST SPLIT
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=42
)

# ---------------------------------------------------
# MODEL
# ---------------------------------------------------

model = DecisionTreeRegressor(
    max_depth=max_depth,
    splitter=splitter,
    random_state=42
)

# TRAIN MODEL
model.fit(X_train, y_train)

# PREDICTIONS
y_pred = model.predict(X_test)

# ---------------------------------------------------
# METRICS
# ---------------------------------------------------

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("📊 Evaluation Metrics")

st.write(f"MAE : {mae:.2f}")
st.write(f"MSE : {mse:.2f}")
st.write(f"R² Score : {r2:.2f}")

# ---------------------------------------------------
# ACTUAL VS PREDICTED
# ---------------------------------------------------

st.subheader("📈 Actual vs Predicted")

results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

st.dataframe(results.head(10))

# ---------------------------------------------------
# SCATTER PLOT
# ---------------------------------------------------

st.subheader("📉 Regression Plot")

fig, ax = plt.subplots()

ax.scatter(X_test, y_test, label="Actual")
ax.scatter(X_test, y_pred, label="Predicted")

ax.set_xlabel("Feature")
ax.set_ylabel("Target")

ax.legend()

st.pyplot(fig)

# ---------------------------------------------------
# TREE VISUALIZATION
# ---------------------------------------------------

st.subheader("🌳 Decision Tree Visualization")

fig2, ax2 = plt.subplots(figsize=(15, 10))

plot_tree(
    model,
    filled=True,
    feature_names=["Feature"],
    ax=ax2
)

st.pyplot(fig2)

# ---------------------------------------------------
# DATASET PREVIEW
# ---------------------------------------------------

st.subheader("📋 Dataset Preview")

st.dataframe(df.head())