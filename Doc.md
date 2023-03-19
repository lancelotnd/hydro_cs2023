Voici les différentes étapes effectuées dans le code :

* Le jeu de données est chargé à partir d'un fichier parquet et est converti en un fichier csv pour une meilleure manipulation à l'aide de la bibliothèque Pandas.
* Les valeurs manquantes dans les données sont remplacées par la moyenne de chaque colonne.
* Une colonne "time_id" est créée en combinant les colonnes "year" et "week" pour faciliter la manipulation des données temporelles.
* La colonne "time_id" est convertie en un objet datetime pour utiliser les fonctionnalités de manipulation de date et d'heure de Pandas.
* Les variables catégorielles sont transformées en variables binaires.
* Les colonnes de caractéristiques et la colonne cible sont identifiées.
* Les données sont normalisées à l'aide d'un objet MinMaxScaler de la bibliothèque Scikit-learn.
* Les données sont divisées en ensembles d'entraînement et de test.
* Un modèle Random Forest est instancié.
* Un objet GridSearchCV est instancié avec le modèle Random Forest et les paramètres à tester pour trouver les meilleurs paramètres.
* Le meilleur modèle Random Forest est trouvé à l'aide de l'objet GridSearchCV.
* Les modèles Random Forest, Régression linéaire et Support Vector Regressor sont instanciés et entraînés avec les données d'entraînement.
* Les performances des modèles sont évaluées à l'aide des données de test.
* Les performances des modèles sont visualisées à l'aide d'un graphique.
* Un DataFrame est créé pour stocker les prévisions pour chaque semaine de l'année 2023.
* Les caractéristiques pour chaque semaine de l'année 2023 sont créées et normalisées.
* Les prévisions sont générées pour chaque semaine de l'année 2023 à l'aide du modèle Random Forest entraîné et stockées dans le DataFrame de prévisions.

# Choix du modele :

* Le modèle Random Forest est choisi comme le meilleur modèle pour faire des prévisions car il donne les meilleures performances en termes de Mean Squared Error (MSE), Mean Absolute Error (MAE) et R-squared (R2). Le coefficient de détermination (R2) de 0,8712158 indique que le modèle est capable d'expliquer **87,12%** de la variance des données de test.





