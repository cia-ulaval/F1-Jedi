from pathlib import Path


class Config:
    """Configuration unique du pipeline EMG a 3 gestes.

    Tous les chemins sont ancres sur la racine du repo : les scripts donnent
    le meme resultat qu'on les lance depuis la racine, depuis collect/ ou
    depuis collect/code_etienne/.
    """

    def __init__(self):
        self.REPO_ROOT = Path(__file__).resolve().parents[2]
        self.ROOT = str(self.REPO_ROOT)

        # ------------------------------------------------------------ gestes
        # Ordre impose : [repos, geste A, geste B].
        # Choix fait sur les donnees S1 et S2 : avec le seul canal EMG du
        # BioPoint, Wrist_Flexion est indiscernable du repos (meme RMS) et
        # Hand_Open change trop d'une session a l'autre. Le trio ci-dessous
        # est le seul qui reste separable d'une session a l'autre.
        self.GESTURES = ["No_Motion", "Hand_Close", "Wrist_Extension"]
        # Identifiants libemg (images/gesture_list.json) pour telecharger les images.
        self.GESTURE_IDS = {"No_Motion": 1, "Hand_Close": 2, "Wrist_Extension": 5}
        self.CLASSES = [self.GESTURE_IDS[name] for name in self.GESTURES]
        self.NUM_CLASSES = len(self.GESTURES)

        # ---------------------------------------------------------- collecte
        self.SESSION_COLLECT = 3        # prochaine session a enregistrer -> dataset/S3
        self.REP_TIME = 5
        self.REST_TIME = 2
        self.NUM_REPS = 7
        self.REPS = list(range(self.NUM_REPS))

        # ------------------------------------------------------ entrainement
        # Sessions utilisees par defaut. Apres une nouvelle collecte, lance
        # plutot : trainning_etienne.py --sessions 3
        self.SESSIONS_TRAIN = [1, 2]
        self.TRAIN_REPS = [2, 0, 4, 1, 5]
        self.TEST_REPS = [3, 6]
        self.N_LAGS = 9                 # nb de retards d'autocorrelation (etage 2)

        # -------------------------------------------------------- temps reel
        self.VOTE_WINDOW = 9            # vote majoritaire sur 9 fenetres (~180 ms) ; 1 = desactive
        self.ADAPT = True               # suivi de la derive (repos, amplitude, spectre)
        self.REST_CALIBRATION_S = 3.0   # mesure du repos au demarrage
        self.RECALIBRATION_S = 3.0      # duree par geste pour la recalibration (touche R)
        self.DELAY_BETWEEN_PREDICTIONS = 0.005  # periode de scrutation du capteur (s)
        self.STALL_TIMEOUT = 0.5        # sans donnees pendant ce delai -> sortie neutre
        # Attente max du premier echantillon. Mesure sur capteur : connexion en
        # 9-10 s apres l'ouverture de la memoire partagee quand le 1er essai BLE
        # reussit, plus quand il echoue ("Retrying"). 10 s ne laissait aucune marge.
        self.TIMEOUT = 30.0

        # --------------------------------------------------------------- jeu
        # Action de chaque geste en mode conduite : "straight", "steer_left",
        # "steer_right" ou "brake". Echange gauche/droite ici si besoin.
        self.GAME_ACTIONS = {
            "No_Motion": "straight",
            "Hand_Close": "steer_left",
            "Wrist_Extension": "steer_right",
        }
        self.STEER_VALUE = 20000        # amplitude du joystick (max 32767)
        self.ACCELERATE_IN_DRIVE = True  # bouton A maintenu en mode conduite

        # ------------------------------------------------------------ chemins
        self.GUI_WIDTH = 700
        self.GUI_HEIGHT = 725
        self.MEDIA_SOURCE = self.REPO_ROOT / "images"              # toutes les images libemg
        self.MEDIA_FOLDER = self.REPO_ROOT / "images" / "sgt_3_gestes"  # seulement les 3 gestes (genere)
        self.DATA_PRE_PATH = self.REPO_ROOT / "dataset"
        self.MODEL_PATH = self.REPO_ROOT / "models"
        self.MODEL_PREFIX = "f1jedi_3g"

        # ---------------------------------------------------------- fenetres
        self.WINDOW_SIZE_MS = 200
        self.WINDOW_INC_MS = 20

        # ------------------------------------------------------------ capteur
        self.EMG_FS = 2000
        self.NOTCH = 60                 # filtre coupe-bande du BioPoint (50 en Europe)
        self.DEVICE_NAME = "BioPoint_v1_3"
        # Veille profonde a l'arret (economise la batterie) : il faut ensuite appuyer
        # sur le bouton du BioPoint pour le reconnecter. False pendant les tests.
        self.SLEEP_SENSOR_ON_EXIT = False

    @property
    def WINDOW_SIZE(self):
        return int(self.WINDOW_SIZE_MS * self.EMG_FS / 1000)

    @property
    def WINDOW_INC(self):
        return int(self.WINDOW_INC_MS * self.EMG_FS / 1000)

    def session_dir(self, session):
        return self.DATA_PRE_PATH / f"S{session}"
