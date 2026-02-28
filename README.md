# NRVR — Neural Real-Time Video Renderer

Système de rendu de jeux vidéo en temps réel par intelligence artificielle générative.

## Concept

Au lieu d'utiliser des millions de polygones, ce système utilise un 
Graphe de Scène Sémantique (GSS) ultra-léger qui décrit la scène 
en quelques centaines d'octets. Un réseau de neurones génère ensuite 
l'image finale en temps réel.

## Comment ça marche
```
Éditeur → GSS (quelques octets) → Modèle IA → Image rendue
```

## Demo live

http://51.255.195.70:8000/game

## Technologies

- Python / PyTorch
- FastAPI
- CNN Neural Renderer
- Graphe de Scène Sémantique (GSS)

## Statut

Prototype fonctionnel — en développement actif.

## Auteur

jomira2903 — 2026
