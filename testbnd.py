import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Charger le jeu de données Spambase (remplacez le chemin par votre propre fichier CSV)

def load_data():
    data = pd.read_csv("spambase_csv.csv" , encoding="ISO-8859-1")
    return data



# Entraîner le modèle XGBoost

def train_xgb_model(X_train, y_train):
    model = XGBClassifier()
    model.fit(X_train, y_train)
    return model

# Interface utilisateur Streamlit
st.title("Détection de Spam avec XGBoost et Streamlit")


# Charger les données
data = load_data()

if st.sidebar.checkbox('Afficher la base de données', False):
        st.subheader("Quelques données du dataset")
        st.write(data.head())
        st.subheader("Description")
        st.write(data.describe())


# Diviser les données en ensemble d'entraînement et ensemble de test
X_train, X_test, y_train, y_test = train_test_split(data.drop('class', axis=1), data['class'], test_size=0.2, random_state=42)

# Entraîner le modèle XGBoost
xgb_model = train_xgb_model(X_train, y_train)

# Ajouter un widget interactif pour la saisie de texte


# Évaluer la précision du modèle sur l'ensemble de test
y_pred_test = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_test)
st.write(f"Précision sur l'ensemble de test : {accuracy:.2f}")

# Afficher un rapport de classification
st.write("Rapport de classification sur l'ensemble de test:")
st.text(classification_report(y_test, y_pred_test))
