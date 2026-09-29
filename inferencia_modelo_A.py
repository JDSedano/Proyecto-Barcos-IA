
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH=Path("resultados_modelo_A/modelo_barcos_A.keras")
model=tf.keras.models.load_model(MODEL_PATH)

def predict_image(path,threshold=.5):
    img=Image.open(path).convert("RGB").resize((80,80))
    x=np.asarray(img,dtype="float32")/255.0
    p=float(model.predict(np.expand_dims(x,0),verbose=0)[0][0])
    label="BARCO" if p>=threshold else "NO BARCO"
    confidence=p if p>=threshold else 1-p
    return label,confidence,p

if __name__=="__main__":
    path=r"C:\Users\roesm\Desktop\Barcos\Exam\imagen.png"
    label,conf,p=predict_image(path)
    print(f"Resultado: {label}")
    print(f"Confianza: {conf*100:.2f}%")
    print(f"Probabilidad de barco: {p*100:.2f}%")
