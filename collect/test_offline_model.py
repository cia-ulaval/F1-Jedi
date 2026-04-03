from offline_model import load_model, predict
import numpy as np
import sys


model_name = sys.argv[1] if len(sys.argv) > 1 else "svm"
model = load_model(model_name=model_name)

feature_count = int(getattr(model, "n_features_in_", 11))
x_new = np.zeros(feature_count, dtype=np.float32)
pred, proba = predict(model, x_new)

print("Modele:", model_name)
print("Classe prédite:", pred)
print("Probabilités:", proba)
