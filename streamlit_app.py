import numpy as np  # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
# Import data 
data = pd.read_excel(r'C:\Users\Mouna Ammali\Desktop\Test_Irly\Data.xlsx')
# Titre de l'application
st.title("Initial Dataset")

# Afficher le DataFrame dans une table interactive
st.dataframe(data)  # Tableau interactif avec des options de tri
## Data viz 
# Titre de l'application
st.title("Data visualisation")
data.hist(figsize=(20, 15), bins=20)
plt.show()
st.pyplot(plt)

data = data.drop(['FirstName','FamilyName','StudentID'],axis = 1 )
# Endoning data 
data_one_hot_encoded = pd.get_dummies(data, columns=['sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason',
       'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
       'higher', 'internet', 'romantic'], drop_first=True) 
data_one_hot_encoded = data_one_hot_encoded.astype(int)


# Separate features from target 
X = data_one_hot_encoded.drop(['FinalGrade'],  axis = 1 )
Y = data_one_hot_encoded['FinalGrade']

# Random Forest Regression 

# Spliting data into test, and train data 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Training the model 
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)

# Evaluating the model 
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Erreur Quadratique Moyenne (MSE) : {mse}")
print(f"Coefficient de Détermination (R²) : {r2}")

importances = rf_model.feature_importances_
feature_names = X.columns
feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)

print("\nImportances des caractéristiques :")
print(feature_importances)



# After evaluationg the features, we have selected the most important features, and the ones that are actionable (meaning that the minstery of Education can have an action on, or can be changed )
"""
Multiple models were tested : linear regression lasso ridge correction, and a correlation map was dressed as well as a PCA to evaluate the link between features 
Trivial observations were made such as that Mother education is highly correlated to father education (implying that educated people tend to marry each other), or that the parents function has a cretain impact on the final grade 
Or that the alchool consupmtion on weekends can affect the consumption during workdays 
"""
y_pred = rf_model.predict(X_test)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6, label="Prédictions")

min_val = min(min(y_test), min(y_pred))  # Valeur minimale pour les axes
max_val = max(max(y_test), max(y_pred))  # Valeur maximale pour les axes
# Calculer l'improvement score (seulement pour les cas où la note prédite > note réelle)
improvement_score = np.where(y_pred > y_test, y_pred - y_test, 0)

# Filtrer les données pour ne garder que celles où l'improvement score est positif (note prédite > note réelle)
positive_improvement_idx = improvement_score > 0
y_test_positive = y_test[positive_improvement_idx]
improvement_score_positive = improvement_score[positive_improvement_idx]

plt.figure(figsize=(8, 6))
plt.scatter(y_test_positive, improvement_score_positive, color='green', alpha=0.6, label="Improvement Score")

plt.title("Improvement Score pour les Notes Prédites > Notes Réelles")
plt.xlabel("Notes Réelles")
plt.ylabel("Improvement Score (Prédiction - Réelle)")

plt.legend()
plt.show()
st.pyplot(plt)

