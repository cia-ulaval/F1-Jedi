# Protocole De Collecte EMG

## Objectif
Construire un dataset offline propre pour entrainer et valider les modeles EMG entre sessions.

## Ordre recommande
1. Collecter `S01`
2. Collecter `S02`
3. Collecter `S03`
4. Entrainer avec `collect/code_etienne/trainning_etienne.py`
5. Valider en temps reel avec `collect/code_etienne/realtime_test_etienne.py`

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
Collecte avec le script courant:

```powershell
.\.venv\Scripts\python.exe .\collect\code_etienne\collect_etienne.py
```

Entrainement sur `S1` avec 5 repetitions pour train et 2 pour test:

```powershell
.\.venv\Scripts\python.exe .\collect\code_etienne\trainning_etienne.py --session 1 --train-reps 2,0,4,1,5 --test-reps 3,6
```

Autre exemple avec un split different dans la meme session:

```powershell
.\.venv\Scripts\python.exe .\collect\code_etienne\trainning_etienne.py --session 1 --train-reps 0,1,2,3,4 --test-reps 5,6
```

Test temps reel avec le dernier modele sauvegarde:

```powershell
.\.venv\Scripts\python.exe .\collect\code_etienne\realtime_test_etienne.py
```

Test temps reel avec un modele explicite:

```powershell
.\.venv\Scripts\python.exe .\collect\code_etienne\realtime_test_etienne.py --model_path .\models\libemg_lda_S1_20260404_015612.pkl
```

Notes:
- Le workflow courant utilise les scripts dans `collect/code_etienne`.
- Les 5 classes sont conservees pour l'entrainement.
- Les options `--train-reps` et `--test-reps` servent a choisir quelles repetitions d'une session vont au train et au test.
- Si aucune repetition n'est fournie, le script prend les valeurs definies dans `config.py`.

## Regle Pratique
Ne passez pas au temps reel tant que la validation par session n'est pas suffisamment stable sur plusieurs collectes.
