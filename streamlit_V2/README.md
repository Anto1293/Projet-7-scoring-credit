# Interface V2 - Tableau de bord scoring client

Cette interface Streamlit V2 a été développée pour permettre une interprétation plus intuitive et accessible des scores de crédit attribués aux clients. 

---

## 🎯 Objectifs

- Visualiser le **score de crédit** d’un client et sa **probabilité associée** via une jauge colorée.
- Afficher une **interprétation du score** grâce à l’importance des variables locales et globales.
- Comparer les données d’un client à **l’ensemble des clients** ou à un **groupe de clients similaires** (par tranche d’âge).
- Permettre une **analyse bi-variée personnalisée** entre deux variables sélectionnées.
- Rendre l’interface **accessible selon les critères WCAG** (graphismes lisibles, noms des variables compréhensibles, contraste, couleurs adaptées).

---

## 🖥️ Fonctionnalités principales

### 🔍 Interface client
- Score de crédit affiché sous forme de **jauge visuelle**.
- Explication locale du score (SHAP / feature importance).
- Informations descriptives du client.

### 🌍 Interface globale
- Affichage des **features les plus influentes globalement**.

### 📊 Graphiques comparatifs
- Graphique comparant la situation d'un client en fonction du panel client entier et du groupe d'âge
- Graphique d'analyse bivariée via sélection des variables

---

## ♿ Accessibilité
- Utilisation de la palette `seaborn-colorblind`.
- Taille des polices, légendes visibles et compréhensibles, contraste amélioré.

---

## 🌐 Déploiement sur Render
Lien vers l'interface en ligne: [https://projet-7-interface-version2.onrender.com]
