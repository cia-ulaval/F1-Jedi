
class Config:
    def __init__(self):
        self.ROOT = "."

        # Data collection
        self.SESSION_COLLECT = 2
        self.CLASSES = [1,2,3,5,4]
        self.NUM_CLASSES = len(self.CLASSES)
        self.REP_TIME = 5
        self.REST_TIME = 2
        self.NUM_REPS = 7
        self.REPS = [i for i in range(0, self.NUM_REPS)]

        # Data split
        self.SESSION_TRAIN = 1
        self.TRAIN_REPS = [2,0,4,1,5]
        self.TEST_REPS = [3,6]
        
        # realtime testing
        self.MAJORITY_VOTE_WINDOW = 40  # Number of samples for majority vote (set to 1 to disable)
        self.DELAY_BETWEEN_PREDICTIONS = 0.02  # seconds
        self.USE_FILTERS = True  # Set to False to disable online filters
        self.TIMEOUT = 10.0  # seconds

        # SGT
        self.GUI_WIDTH = 700
        self.GUI_HEIGHT = 725
        self.MEDIA_FOLDER = self.ROOT + "/images/"
        self.DATA_PRE_PATH = self.ROOT + "/dataset/"
        self.DATA_PATH_SUBJECT = str(self.DATA_PRE_PATH) + "S" + str(self.SESSION_COLLECT) + "/"
        self.DATA_PATH_TRAIN = str(self.DATA_PRE_PATH)
        self.MODEL_PATH = self.ROOT + "/../../models/"
        
        # Windows in ms
        self.WINDOW_SIZE_MS = 200
        self.WINDOW_INC_MS = 20

        # sampling frequencies
        self.PPG_FS = 50
        self.EMG_FS = 2000
        self.ECG_FS = 500
        self.EDA_FS = 50
        self.IMU_FS = 100
        self.TEMP_FS = 0.5
        self.TEMP_SIMULATED_FS = 2000
        self.NOTCH = 60