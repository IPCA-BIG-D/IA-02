import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Declarar colunas do dataset
column_names = ["class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor", "gill-attachment",
                "gill-spacing", "gill-size", "gill-color", "stalk-shape", "stalk-root", "stalk-surface-above-ring",
                "stalk-surface-below-ring", "stalk-color-above-ring", "stalk-color-below-ring", "veil-type",
                "veil-color", "ring-number", "ring-type", "spore-print-color", "population", "habitat"]

# Lê dados do dataset
mushroom_data = pd.read_csv('./input/mushrooms.csv', header=None, names=column_names)

# Encode categorical features
le = LabelEncoder()
for column in mushroom_data.columns:
    mushroom_data[column] = le.fit_transform(mushroom_data[column])

# Separar os dados em features (X) e objetivo (y)
X = mushroom_data.drop('class', axis=1)
y = mushroom_data['class']

# Dividir os dados em dataset de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------------------- Decision Tree  --------------------------------------------

# Decision Tree Classifier
# dt_clf = DecisionTreeClassifier(random_state=42)
dt_clf = DecisionTreeClassifier(max_depth=5, min_samples_split=2, min_samples_leaf=1, random_state=42) # amostragem
dt_clf.fit(X_train, y_train)
dt_y_pred = dt_clf.predict(X_test)

# Avalia a Decision Tree classifier
dt_accuracy = accuracy_score(y_test, dt_y_pred)
dt_conf_matrix = confusion_matrix(y_test, dt_y_pred)
dt_class_report = classification_report(y_test, dt_y_pred)

# Apresenta resultados Decision Tree
print("Decision Tree Classifier Results:")
print("Accuracy:", dt_accuracy)
print("Confusion Matrix:\n", dt_conf_matrix)
print("Classification Report:\n", dt_class_report)

# -------------------------------------------------------------------------------------------------

# -------------------------------------------- RandomForest --------------------------------------------

# RandomForest Classifier
# rf_clf = RandomForestClassifier(random_state=42)
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=2, min_samples_leaf=1, random_state=42) # amostragem
rf_clf.fit(X_train, y_train)
rf_y_pred = rf_clf.predict(X_test)
# Avalia o RandomForest classifier
rf_accuracy = accuracy_score(y_test, rf_y_pred)
rf_conf_matrix = confusion_matrix(y_test, rf_y_pred)
rf_class_report = classification_report(y_test, rf_y_pred)

# Apresenta resultados RandomForest
print("\nRandomForest Classifier Results:")
print("Accuracy:", rf_accuracy)
print("Confusion Matrix:\n", rf_conf_matrix)
print("Classification Report:\n", rf_class_report)

# -------------------------------------------------------------------------------------------------

# -------------------------------------------- K-means --------------------------------------------

# K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=42) # amostragem
X_train_clustered = kmeans.fit_predict(X_train)

# Usa K-Means para testar o conjunto 
X_test_clustered = kmeans.predict(X_test)

# Avalia K-Means clusters
kmeans_accuracy = accuracy_score(y_test, X_test_clustered)
kmeans_conf_matrix = confusion_matrix(y_test, X_test_clustered)

# Apresenta resultados K-Means clustering
print("\nK-Means Clustering Results:")
print("Accuracy:", kmeans_accuracy)
print("Confusion Matrix:\n", kmeans_conf_matrix)

# -------------------------------------------------------------------------------------------------

# Graphs region

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(18, 10))

# Decision Tree Matriz Confusão 
sns.heatmap(dt_conf_matrix, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
axes[0, 0].set_title('Decision Tree Confusion Matrix')

# RandomForest Matriz Confusão 
sns.heatmap(rf_conf_matrix, annot=True, fmt='d', cmap='Greens', ax=axes[0, 1])
axes[0, 1].set_title('RandomForest Confusion Matrix')

# K-Means Matriz Confusão 
sns.heatmap(confusion_matrix(y_test, X_test_clustered), annot=True, fmt='d', cmap='Reds', ax=axes[1, 0])
axes[1, 0].set_title('K-Means Confusion Matrix')

# Scatter plot for K-Means Clusters (utilizando as 2 primeiras features(X)) com uma amostragem de "sample_size"
sample_size = 1000

sample_indices = np.random.choice(X_test.shape[0], sample_size, replace=False)
axes[1, 1].scatter(X_test.iloc[sample_indices, 0], X_test.iloc[sample_indices, 1], c=X_test_clustered[sample_indices], cmap='viridis')
axes[1, 1].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='X', s=200, c='red', label='Centroids')
axes[1, 1].set_title(f'K-Means Clustering (Sampled {sample_size} points)')
axes[1, 1].legend()

plt.tight_layout()
plt.show()