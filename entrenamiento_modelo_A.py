
import re, json
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, RandomFlip, RandomRotation, RandomZoom, RandomContrast
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import load_img, img_to_array

SEED=42; np.random.seed(SEED); tf.random.set_seed(SEED)
IMG_SIZE=(80,80); BATCH_SIZE=32; EPOCHS=40; LR=0.001; TEST_SIZE=0.10
IMAGE_DIR=Path(r"C:\Users\juand\Downloads\Barcos\Barcos\archive\shipsnet\shipsnet")
OUT=Path("resultados_modelo_A"); OUT.mkdir(exist_ok=True)
MODEL_PATH=OUT/"modelo_barcos_A.keras"

if not IMAGE_DIR.exists(): raise FileNotFoundError(f"No existe IMAGE_DIR: {IMAGE_DIR}")

data=[]; pattern=r'^(\d)__(.+?)__(-?\d+\.\d+)_(\d+\.\d+)\.png$'
for p in sorted(IMAGE_DIR.glob("*.png")):
    m=re.match(pattern,p.name)
    if m and int(m.group(1)) in (0,1):
        data.append({"filename":p.name,"filepath":str(p),"label":int(m.group(1))})
df=pd.DataFrame(data)
if df.empty: raise RuntimeError("No se encontraron imágenes con el formato esperado.")

print(f"Total: {len(df)}"); print(df.label.value_counts().sort_index())

X=np.asarray([img_to_array(load_img(p,color_mode="rgb",target_size=IMG_SIZE)) for p in df.filepath],dtype="float32")/255.0
y=df.label.to_numpy(dtype="int32")
X_trainval,X_test,y_trainval,y_test=train_test_split(X,y,test_size=TEST_SIZE,stratify=y,random_state=SEED)

augmentation=Sequential([
    RandomFlip("horizontal"), RandomRotation(0.05),
    RandomZoom(0.10), RandomContrast(0.10)
],name="augmentation")

def create_model_A():
    m=Sequential([
        Input(shape=(80,80,3)), augmentation,
        Conv2D(32,3,padding="same",activation="relu"), BatchNormalization(), MaxPooling2D(),
        Conv2D(64,3,padding="same",activation="relu"), BatchNormalization(), MaxPooling2D(),
        Conv2D(128,3,padding="same",activation="relu"), BatchNormalization(), MaxPooling2D(),
        Conv2D(256,3,padding="same",activation="relu"), BatchNormalization(), MaxPooling2D(),
        Flatten(), Dense(128,activation="relu"), Dropout(.5),
        Dense(1,activation="sigmoid")
    ])
    m.compile(optimizer=Adam(LR),loss="binary_crossentropy",metrics=["accuracy"])
    return m

def callbacks():
    return [
        EarlyStopping(monitor="val_loss",patience=6,restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss",factor=.5,patience=3,min_lr=1e-6)
    ]

# 5-fold CV: test is never used here.
rows=[]
skf=StratifiedKFold(5,shuffle=True,random_state=SEED)
for fold,(tr,va) in enumerate(skf.split(X_trainval,y_trainval),1):
    print(f"\nFOLD {fold}/5")
    m=create_model_A()
    m.fit(X_trainval[tr],y_trainval[tr],validation_data=(X_trainval[va],y_trainval[va]),
          epochs=EPOCHS,batch_size=BATCH_SIZE,verbose=0,callbacks=callbacks())
    prob=m.predict(X_trainval[va],verbose=0).ravel()
    pred=(prob>=.5).astype(int)
    rows.append([fold,accuracy_score(y_trainval[va],pred),
                 precision_score(y_trainval[va],pred,zero_division=0),
                 recall_score(y_trainval[va],pred,zero_division=0),
                 f1_score(y_trainval[va],pred,zero_division=0)])
    print(f"Accuracy: {rows[-1][1]*100:.2f}%")

cv=pd.DataFrame(rows,columns=["fold","accuracy","precision","recall","f1"])
cv.to_csv(OUT/"resultados_validacion_cruzada.csv",index=False)
print("\nCV promedio:")
print(cv.mean(numeric_only=True)*100)

# Final training. The 10% test remains untouched.
model=create_model_A()
history=model.fit(X_trainval,y_trainval,validation_split=.10,epochs=EPOCHS,batch_size=BATCH_SIZE,
                  shuffle=True,verbose=1,callbacks=callbacks())
model.save(MODEL_PATH)

prob=model.predict(X_test,batch_size=BATCH_SIZE,verbose=0).ravel()
pred=(prob>=.5).astype(int)
acc=accuracy_score(y_test,pred); prec=precision_score(y_test,pred,zero_division=0)
rec=recall_score(y_test,pred,zero_division=0); f1=f1_score(y_test,pred,zero_division=0)
cm=confusion_matrix(y_test,pred,labels=[0,1])

print("\n=== TEST FINAL ===")
print(f"Accuracy : {acc*100:.2f}%")
print(f"Precision: {prec*100:.2f}%")
print(f"Recall   : {rec*100:.2f}%")
print(f"F1       : {f1*100:.2f}%")
print("Confusion matrix:\n",cm)
print(classification_report(y_test,pred,target_names=["No barco","Barco"],digits=4,zero_division=0))

np.savetxt(OUT/"matriz_confusion.csv",cm,delimiter=",",fmt="%d")
with open(OUT/"metricas_test.json","w",encoding="utf8") as f:
    json.dump({"accuracy":float(acc),"precision":float(prec),"recall":float(rec),"f1":float(f1),
               "test_samples":len(y_test),"threshold":.5},f,indent=2)

plt.figure(); plt.plot(history.history["accuracy"],label="Train"); plt.plot(history.history["val_accuracy"],label="Validation")
plt.xlabel("Época"); plt.ylabel("Accuracy"); plt.title("Accuracy"); plt.legend(); plt.grid(); plt.tight_layout()
plt.savefig(OUT/"accuracy_entrenamiento.png",dpi=150); plt.close()

plt.figure(); plt.plot(history.history["loss"],label="Train"); plt.plot(history.history["val_loss"],label="Validation")
plt.xlabel("Época"); plt.ylabel("Loss"); plt.title("Loss"); plt.legend(); plt.grid(); plt.tight_layout()
plt.savefig(OUT/"loss_entrenamiento.png",dpi=150); plt.close()

plt.figure(); plt.imshow(cm); plt.title("Matriz de confusión - Test"); plt.xlabel("Predicción"); plt.ylabel("Real")
plt.xticks([0,1],["No barco","Barco"]); plt.yticks([0,1],["No barco","Barco"])
for i in range(2):
    for j in range(2): plt.text(j,i,str(cm[i,j]),ha="center",va="center")
plt.tight_layout(); plt.savefig(OUT/"matriz_confusion.png",dpi=150); plt.close()

print(f"\nModelo guardado: {MODEL_PATH.resolve()}")
print("META 98%:", "CUMPLIDA" if acc>=.98 else "NO CUMPLIDA; requiere optimización adicional.")
