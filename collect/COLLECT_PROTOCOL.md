# Protocole De Collecte EMG

## Objectif
Construire un dataset offline propre pour entrainer et valider le LDA sur plusieurs sujets.

## Ordre recommande
1. Collecter `S01`
2. Collecter `S02`
3. Collecter `S03`
4. Entrainer avec `collect/lda_offline.py`

## Avant Chaque Session
- Positionner le capteur toujours au meme endroit.
- Garder la meme main, la meme posture du bras et du poignet.
- Verifier que le signal EMG arrive bien avant de commencer.
- Faire une courte contraction de test pour voir si le capteur reagit.

## Pendant La Collecte
- Garder chaque geste stable pendant toute la pose.
- Eviter les mouvements parasites du bras, de l'epaule et du tronc.
- Relacher completement pendant les pauses.
- Si un geste est rate, supprimer le dossier du sujet et recommencer proprement.

## Gestes
- `Hand_Open`
- `Hand_Close`
- `No_Motion`
- `Wrist_Extension`
- `Wrist_Flexion`

## Commandes
Collecte auto sur le prochain sujet libre:

```powershell
.\.venv\Scripts\python.exe .\collect\collect.py
```

Collecte explicite pour `S02`:

```powershell
.\.venv\Scripts\python.exe .\collect\collect.py --subject S02
```

Collecte explicite pour `S03`:

```powershell
.\.venv\Scripts\python.exe .\collect\collect.py --subject S03
```

Collecte de plusieurs repetitions consecutives, chacune en tant que sujet distinct:

```powershell
.\.venv\Scripts\python.exe .\collect\collect.py --repetitions 3
```

Collecte de plusieurs repetitions en commencant a `S10`:

```powershell
.\.venv\Scripts\python.exe .\collect\collect.py --subject S10 --repetitions 3
```

Entrainement offline:

```powershell
.\.venv\Scripts\python.exe .\collect\lda_offline.py
```

## Regle Pratique
Ne passez pas au temps reel tant que la validation par sujet n'est pas disponible avec au moins `S01`, `S02` et `S03`.
