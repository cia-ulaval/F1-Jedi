# Protocole de collecte EMG (3 gestes)

## Objectif
Enregistrer une session propre, entrainer le modele a 3 gestes, puis jouer.

## Les 3 gestes
| Geste | Role en jeu | Pourquoi |
|-------|-------------|----------|
| `No_Motion` | tout droit | main detendue |
| `Hand_Close` | gauche | poing ferme, contraction moyenne |
| `Wrist_Extension` | droite | poignet releve, contraction forte |

Le BioPoint n'a qu'un canal EMG. Sur les sessions S1 et S2, `Wrist_Flexion` a le meme
niveau que le repos et `Hand_Open` change trop d'une session a l'autre : ils ne sont plus
utilises. Les anciennes sessions a 5 gestes restent lisibles, le script n'en garde que les
3 gestes ci-dessus.

## Avant chaque session
- Capteur toujours au meme endroit et sur le meme bras. Bonne position : celle ou
  `Wrist_Extension` donne un signal nettement plus fort que le repos (verifier avec
  realtime_test_etienne.py, colonne rms).
- Peau propre et seche. Attendre 1 a 2 minutes apres la pose : le contact se stabilise
  quand la peau se rechauffe sous l'electrode.
- Meme posture du bras qu'en jeu.

## Pendant la collecte
- Tenir chaque geste stable pendant toute la pose, avec la force utilisee en jeu
  (moderee : une contraction maximale fatigue vite et ne ressemble pas au jeu).
- Relacher completement pendant les pauses.
- Si une repetition est ratee, utiliser le bouton "Redo" du GUI.

## Commandes
```powershell
# 1. Collecte (3 gestes x 7 repetitions de 5 s, environ 3 minutes)
python .\collect\code_etienne\collect_etienne.py --session 3

# 2. Entrainement + evaluation (train reps 2,0,4,1,5 ; test reps 3,6 par defaut)
python .\collect\code_etienne\trainning_etienne.py --sessions 3

# Variantes
python .\collect\code_etienne\trainning_etienne.py --sessions 3 --train-reps 0,1,2,3,4 --test-reps 5,6
python .\collect\code_etienne\trainning_etienne.py --sessions 1,2,3            # plusieurs sessions
python .\collect\code_etienne\trainning_etienne.py --sessions 2 --test-sessions 3   # test inter-session

# 3. Test temps reel (dernier modele charge automatiquement)
python .\collect\code_etienne\realtime_test_etienne.py

# 4. Jeu
python .\collect\game.py
```

Sans capteur :
```powershell
python .\collect\code_etienne\realtime_test_etienne.py --replay 2
python -m pytest .\collect\code_etienne\tests -q
```

## Lire le resultat de l'entrainement
- "Exactitude temps reel simule" : les repetitions de test rejouees dans le decodeur du
  jeu. Viser 95 % ou plus. Sous 90 %, replacer le capteur et recollecter.
- "rapport" repos/actif : viser 5x ou plus. Un rapport faible veut dire que le capteur
  n'est pas sur le bon muscle ou que les gestes sont trop mous.

## Quand le decodage se degrade en jeu
Au bout de quelques minutes, la sueur, la temperature de la peau et la fatigue changent le
signal. Le decodeur suit ces changements tout seul (niveau de repos, amplitude, spectre).
Si ca ne suffit plus :
1. Appuyer sur `R` dans le pont de jeu : recalibration guidee de 15 s (repos, puis chaque
   geste 3 s). La manette est relachee pendant ce temps.
2. Si la recalibration est refusee ou ne suffit pas : essuyer la peau, replacer le capteur,
   refaire une collecte courte.

## Reglages (collect/code_etienne/config.py)
- `GESTURES`, `GAME_ACTIONS` : gestes et actions de jeu (inverser gauche/droite ici).
- `VOTE_WINDOW` : 9 fenetres (180 ms). Plus grand = plus stable mais plus lent.
- `ADAPT` : suivi de derive (laisser a True).
- `NOTCH` : 60 Hz en Amerique du Nord.
