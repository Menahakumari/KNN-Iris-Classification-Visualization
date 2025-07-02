# KNN-Iris-Classification-Visualization
🌸 KNN Iris Classification & Visualization
A simple machine learning project that uses the K‑Nearest Neighbors (KNN) algorithm to classify species in the Iris dataset and visualize performance metrics including accuracy and decision boundaries.

📂 Project Structure

bash

Copy

Edit

KNN-Iris-Classification-Visualization/

├── 
iris.csv                  # Iris dataset (downloaded from Kaggle or UCI)


├──
knn_iris.py               # Python script with the full code


├──
knn_iris.ipynb            # Jupyter Notebook version


├──
README.md                 # This file


└── 
outputs/
    
    
    ├── accuracy_plot.png    # Accuracy vs K values
    
    
    └── decision_boundary.png # Decision boundary visualization


    
🔍 About the Dataset
Dataset: Iris Dataset on Kaggle
https://www.kaggle.com/datasets/uciml/iris
Alternative: UCI Iris Dataset

Classes: Setosa, Versicolor, Virginica

Features:

Sepal Length (cm)

Sepal Width (cm)

Petal Length (cm)

Petal Width (cm)

📊 What This Project Does
Loads and preprocesses the Iris dataset (from CSV)

Encodes class labels using LabelEncoder

Normalizes the features using StandardScaler

Trains KNN classifier with different K values (1 to 10)

Evaluates using:

Accuracy

Confusion Matrix

Visualizes:

Accuracy vs. K plot

Decision boundaries (2D, Petal Length & Width)

🚀 Getting Started
✅ Requirements
Install the dependencies:

bash
Copy
Edit
pip install pandas numpy matplotlib scikit-learn
▶️ Running the Script
bash
Copy
Edit
python knn_iris.py
📓 Or Run the Notebook
Use Jupyter or Google Colab to open and run knn_iris.ipynb.

📈 Visualizations
Accuracy Plot: Shows classification accuracy across different K values

