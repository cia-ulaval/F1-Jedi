from lda import load_model, predict
import numpy as np

model = load_model()

x_new = np.array([0.12, 0.87, 1.04, 0.33], dtype=np.float32)
pred, proba = predict(model, x_new)

print("Classe prédite:", pred)
print("Probabilités:", proba)