# KNN-Iris-Classification-Visualization
# 🌸 KNN Iris Classification & Visualization

A simple machine learning project using the **K‑Nearest Neighbors (KNN)** algorithm to classify species in the classic **Iris dataset**. This project evaluates model accuracy, visualizes confusion matrices, and plots decision boundaries for better understanding.

---

## 📂 Project Structure
KNN-Iris-Classification-Visualization/


├── iris.csv # Iris dataset (downloaded from Kaggle or UCI)


├── knn_iris.py # Python script with the full KNN implementation


├── knn_iris.ipynb # Jupyter Notebook version (optional)


├── README.md # Project overview and instructions


└── outputs/


├── accuracy_plot.png # Accuracy vs K values graph


└── decision_boundary.png # Decision boundary visualization (2D)


---

## 🔍 About the Dataset

- **Dataset Source**: [Kaggle – Iris Dataset](https://www.kaggle.com/datasets/uciml/iris)
- **Original Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris)
- **Classes**:
  - Iris-setosa
  - Iris-versicolor
  - Iris-virginica
- **Features**:
  - Sepal Length (cm)
  - Sepal Width (cm)
  - Petal Length (cm)
  - Petal Width (cm)

---

## 📊 What This Project Does

- Loads the dataset from CSV (Kaggle)
- Encodes class labels using `LabelEncoder`
- Normalizes feature values using `StandardScaler`
- Splits data into training and testing sets
- Trains a `KNeighborsClassifier` with multiple values of **K (1–10)**
- Evaluates performance using:
  - **Accuracy**
  - **Confusion Matrix**
- Plots:
  - **Accuracy vs K graph**
  - **Decision boundaries** using 2D feature space (Petal Length & Width)

---

## 🚀 Getting Started

### ✅ Prerequisites

Make sure Python and the following libraries are installed:

```bash
pip install pandas numpy matplotlib scikit-learn

python knn_iris.py

📓 Or Use the Jupyter Notebook
Open and run the knn_iris.ipynb notebook using Jupyter or Google Colab.


-----
📈 Visual Outputs
📌 Accuracy Plot
Shows model accuracy as K increases (from K=1 to K=10). Helps determine the best K value for classification.

📌 Decision Boundary
Visualizes how KNN separates classes based on two selected features (Petal Length & Petal Width). Useful for understanding model behavior.
