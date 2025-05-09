import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

data = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/decision-tree-project-tutorial/main/diabetes.csv")
datacore = data.copy()
data1 = data.copy()
data.head()

data.describe()

data.info()

#datacore = datacore.drop( )
datacore.drop(labels=datacore[datacore["Glucose"] == 0 ].index, inplace=True)
datacore.drop(labels=datacore[datacore["BloodPressure"] == 0 ].index, inplace=True)
datacore.drop(labels=datacore[datacore["BMI"] == 0 ].index, inplace=True)
datacore.drop(labels=datacore[datacore["SkinThickness"] == 0 ].index, inplace=True)
datacore.drop(labels=datacore[datacore["Insulin"] == 0 ].index, inplace=True)

datacore["Outcome"].value_counts()

# Crear grupos de edad
datacore["AgeGroup"] = pd.cut(datacore["Age"], bins=[20, 30, 40, 50, 60, 70, 80], 
                              labels=["21-30", "31-40", "41-50", "51-60", "61-70", "71-80"])

# Calcular proporción de Outcome = 1 (diabetes) por grupo de edad
diabetes_por_edad = datacore.groupby("AgeGroup")["Outcome"].mean().reset_index()

# Crear gráfico
fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(data=diabetes_por_edad, x="AgeGroup", y="Outcome", color="#ff7f0e", ax=ax)

# Títulos
ax.set_title("Proporción de Diabetes por Grupo de Edad")
ax.set_ylabel("Proporción de Casos de Diabetes")
ax.set_xlabel("Grupo de Edad")

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Variables numéricas
variables = ["Glucose", "BloodPressure", "Insulin", "BMI", "SkinThickness", "Age"]

# Crear subplots
fig, axes = plt.subplots(3, 2, figsize=(18, 12))

# Recorrer variables y ejes
for ax, var in zip(axes.flat, variables):
    sns.histplot(data=datacore, x=var, bins=30, kde=False, color="skyblue", ax=ax)
    ax.set_title(f"Histograma de {var}")
    ax.set_xlabel(var)
    ax.set_ylabel("Frecuencia")

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(3, 2, figsize=(18, 12))

# Lista de variables a graficar
variables = ["Glucose", "BloodPressure", "Insulin", "BMI", "SkinThickness", "Age"]

# Graficar cada variable
for ax, var in zip(axes.flat, variables):
    sns.kdeplot(data=datacore, x=var, hue="Outcome", fill=True, common_norm=False, alpha=0.5, ax=ax)
    ax.set_title(f"Distribución de {var} según Outcome")
    ax.set_xlabel(var)
    ax.set_ylabel("Densidad")

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates

# Eliminar columna no numérica
datacore.drop(columns=["AgeGroup"], inplace=True)

# Gráfico de coordenadas paralelas
plt.figure(figsize=(12, 6))
parallel_coordinates(datacore, "Outcome", color=("#E58139", "#39E581", "#8139E5"))

# Rotar etiquetas de los ejes
plt.xticks(rotation=45)  # Cambia a 90 si quieres más inclinación

# Mostrar gráfico
plt.title("Coordenadas paralelas por clase de Outcome")
plt.tight_layout()
plt.show()

from scipy.stats.mstats import winsorize

datacore['Insulin_winsorized'] = winsorize(datacore['Insulin'], limits=[0.05, 0.05]) 

datacore.columns

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

# Define tus variables independientes (X) y dependiente (y)
X = datacore.drop(columns=['Outcome'], axis=1)  # Reemplaza 'columna_objetivo' por el nombre real de la variable objetivo
y = datacore["Outcome"]

# Divide el dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

# Imprimir tamaños resultantes
print('Tamaño set de entrenamiento: ', X_train.shape, y_train.shape)
print('Tamaño set de prueba: ', X_test.shape, y_test.shape)

# Distribución de categorías
print('Distribución de categorías dataset original: ', y.value_counts(normalize=True))
print('Distribución de categorías dataset entrenamiento: ', y_train.value_counts(normalize=True))
print('Distribución de categorías dataset prueba: ', y_test.value_counts(normalize=True))

# Muestra las primeras filas del conjunto de entrenamiento
print(datacore["Outcome"].value_counts())
print(X.columns)
 
model = DecisionTreeClassifier(random_state=42) 
model.fit(X_train, y_train)

predicciones = model.predict(X_test)
predicciones

matriz = pd.DataFrame(confusion_matrix(y_test, predicciones), columns = ["No tiene diabetes", "Tiene diabetes"], index =["No tiene diabetes", "Tiene diabetes"])
matriz

accuracy_score(y_test, predicciones)

from sklearn.tree import plot_tree

plt.figure(figsize=(30,15))
plot_tree(model, feature_names=X_train.columns, class_names=["No diabetes", "Diabetes"], filled=True, rounded=True,)

plt.savefig("arbol de clasificacion.png", dpi=700)

print("Profundidad de arbol", model.get_depth())
print("Bumero de hojas de arbol", model.get_n_leaves())

model.feature_importances_

from sklearn.model_selection import StratifiedKFold,cross_val_score

skf = StratifiedKFold(shuffle=True, random_state=42)

model = DecisionTreeClassifier(random_state=42) 
scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="f1_weighted")

print("F1-score ponderado entrenamiento", scores.mean())

#Entrenar arbol con un set de entrenamiento
model.fit(X_train,y_train)

#Puntaje de set de prueba
#print("F1-score ponderado prueba", score_test)

matriz = pd.DataFrame(confusion_matrix(y_test, predicciones), columns = ["No tiene diabetes", "Tiene diabetes"], index =["No tiene diabetes", "Tiene diabetes"])
matriz


model = DecisionTreeClassifier(max_depth=5, random_state=42) 
model.fit(X_train, y_train)

plt.figure(figsize=(30,15))
plot_tree(model, feature_names=X_train.columns, class_names=["No diabetes", "Diabetes"], filled=True, rounded=True,)

plt.savefig("arbol de clasificacion dephh 5.png", dpi=700)


# Características del árbol
print('Profundidad del árbol: ', model.get_depth())
print('Número de hojas del árbol: ', model.get_n_leaves())

from sklearn.model_selection import StratifiedKFold,cross_val_score

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

model = DecisionTreeClassifier(random_state=42) 
scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="f1_weighted")

print("F1-score ponderado entrenamiento", scores.mean())

#Entrenar arbol con un set de entrenamiento
model.fit(X_train,y_train)

#Puntaje de set de prueba
#print("F1-score ponderado prueba", score_test)

matriz = pd.DataFrame(confusion_matrix(y_test, predicciones), columns = ["No tiene diabetes", "Tiene diabetes"], index =["No tiene diabetes", "Tiene diabetes"])
matriz

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

# Definir los hiperparámetros a probar
param_grid = {
    'criterion': ['gini', 'entropy'],          # Función de impureza
    'max_depth': [3, 5, 7, 9, None],           # Profundidad del árbol
    'min_samples_split': [2, 5, 10],           # Mínimo de muestras para dividir un nodo
    'min_samples_leaf': [1, 2, 4]              # Mínimo de muestras por hoja
}

# Modelo base
modelo_base = DecisionTreeClassifier(random_state=42)

# Grid Search con validación cruzada de 5 folds
grid_search = GridSearchCV(estimator=modelo_base, 
                           param_grid=param_grid,
                           cv=5, 
                           scoring='accuracy',
                           n_jobs=-1,
                           verbose=1)

# Ajustar a los datos de entrenamiento
grid_search.fit(X_train, y_train)

# Mostrar los mejores parámetros
print("Mejores hiperparámetros:", grid_search.best_params_)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# Entrenar con los mejores hiperparámetros
modelo_final = DecisionTreeClassifier(
    criterion='gini',
    max_depth=7,
    min_samples_leaf=4,
    min_samples_split=10,
    random_state=42
)
modelo_final.fit(X_train, y_train)

# Predicciones
y_pred = modelo_final.predict(X_test)

# Evaluación
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred))
print("F1-score ponderado:", f1_score(y_test, y_pred, average='weighted'))

modelo_final = grid_search.best_estimator_
modelo_final

import joblib

# Guardar el modelo
joblib.dump(modelo_final, 'modelo_arbol_diabetes.pkl')
print("Modelo guardado como 'modelo_arbol_diabetes.pkl'")

modelo_cargado = joblib.load('modelo_arbol_diabetes.pkl')
