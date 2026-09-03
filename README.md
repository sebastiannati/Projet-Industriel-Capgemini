# Projet-Industriel-Capgemini

## Installation des librairies

Pour avoir les mêmes versions que nous il faut lancer le fichier *install_requirements*

## Extraction des coordonnées

Pour extraire les coordonnnées des vidéos il faut éxectuter le fichier *extract_coordinates.py*. 

Ce fichier va ouvrir les vidéos une par une et sauvegarder les coordonnées dans un fichier csv nommé : *video_coordinates.csv* dans le dossier "csv".

Pour éviter de traiter les vidéos plusieurs fois nous nottons les videos traitées dans un fichier *videos_traitees.txt* dans le dossier "videos".

## Création du tableaux des signes de fatigues

Pour créer le tableau des signes de fatigues de chacune des vidéos il faut lancer le fichier *dataframe.py* et enregistre le tableau dans le fichier *signes.py* dans le dossier *csv*

Ce fichier utilise les fichiers du dossier "src", ces fichiers sont en fait les ficihers permettant de calculer les signes de fatigues.

Il enregistre les graphiques des signes dans le dossier "graphics".

## Utiliser le démonstrateur

Pour utiliser le démonstrateur il suffit d'executer le fichier *demonstrateur.py*

## Les modèles

Les modèles entraînés sont dans le dossier *modele_ml*


## Limitations

1. les données sont récoltées qu'a partir de video de personnes fixes devant la caméra et aucunement dans une situation concrete de conduite ou le coportement peut etre bien différent (parler, verifier les angles morts, les retroviseurs, passé les vitesses, tourner le volant)

2. au moment de la realisation du projet nous n'avions pas eu de cours de Machine Learning nous permettant de comprendre les subtilitées des modèles et comment les utiliser au mieux ce qui nous a fait entrainer un RF sur 7000 features temporelles reliées entre elles

3. data leakage, les videos pour les données on ete decoupé sur des videos de 10min ainsi on a plusieurs instances qui traduisent le comportement d'une seule personnes et donc quand on split le jeu en train/test on permet au model d'etre evalué sur des personnes deja vu a l'entrainement

4. Optimisation Notebook not imported but tried Gridsearch and RandomSearch + CV to have the best model, in enhanced with the training model

