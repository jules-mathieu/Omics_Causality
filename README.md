# Caulzalité - Application pour les simulations

## Objectifs du projet
Cette partie vise à générer des données métabolomiques à partir de données génomiques réelles, avec un lien de causalité qui respecte le cadre proposé par Judea Pearl. Le DAG simulé ici est très simple, avec simplement un SNP causal sur une variable latente de metabolites. Le jeu de données de métabolites est ensuite inféré depuis cette VL.
Nous pouvons alors conduire des analyses de causalité entre 1) la VL simulée et l'ensemble du genome (validation all genome). 2) Nous réalisons des test de comparaison des moyennes (ttest) pour récupérer une p-value sur la significativité de l'ATE retrouvé. 
Cela permet notamment de comparer des stratégies de réduction de dimensions en comparant la VL simulée et des VL retrouvées depuis le jeu de données en plus haute dimension.
3) On qualifie l'impact de chaque paramètre du modèle indépendemment par analyse de sensibilité.

## Structure
Les données brutes **colza_genotype.parquet**, présentes dans le dossier ***data*** sont d'abord traitées par **data_cleaning.py**, ce qui permet de produire **colza_genotype_cleaned.parquet**.

L'application, située dans **omics_causality_app.py**, peut être lancée via la commande : *streamlit run omics_causality_app.py*.

## L'application
L'application comporte quatre pages indépendantes, qui reprend toutes les étapes du projet pour la partie simulations.

- Génération des données selon nos hypothèses
- Validation all genome (manhattan plot like)
- Tests de puissance pour retrouver le SNP
- Analyse de sensibilité pour déterminer les drivers de notre modèle
