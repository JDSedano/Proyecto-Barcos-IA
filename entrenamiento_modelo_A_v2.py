# ============================================================
# MODELO A - V2
# CLASIFICACIÓN DE IMÁGENES: BARCO / NO BARCO
# ============================================================
#
# Mejoras respecto al modelo original:
#   1. Data Augmentation más completo y controlado
#   2. GlobalAveragePooling2D en lugar de Flatten
#   3. SpatialDropout2D
#   4. AdamW con learning rate menor
#   5. EarlyStopping
#   6. ReduceLROnPlateau
#   7. ModelCheckpoint
#   8. Validación cruzada 5-fold
#   9. Búsqueda del mejor umbral usando VALIDACIÓN
#  10. Test final completamente separado
#  11. Guardado de métricas, matrices y gráficas
# ============================================================

import re
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Dense,
    Dropout,
    BatchNormalization,
    SpatialDropout2D,
    GlobalAveragePooling2D,
    RandomFlip,
    RandomRotation,
    RandomZoom,
    RandomContrast,
    RandomTranslation,
    RandomBrightness
)

from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

from tensorflow.keras.utils import load_img, img_to_array


# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

IMG_SIZE = (80, 80)

BATCH_SIZE = 32

EPOCHS = 50

LR = 0.0005

TEST_SIZE = 0.10

VAL_SIZE = 0.10

THRESHOLD_DEFAULT = 0.50


# ============================================================
# 2. RUTAS
# ============================================================

IMAGE_DIR = Path(
    r"C:\Users\juand\Downloads\Barcos\Barcos\archive\shipsnet\shipsnet"
)

OUT = Path("resultados_modelo_A_v2")

OUT.mkdir(exist_ok=True)

MODEL_PATH = OUT / "modelo_barcos_A_v2.keras"

BEST_MODEL_PATH = OUT / "mejor_modelo_barcos_A_v2.keras"


# ============================================================
# 3. VERIFICAR DATASET
# ============================================================

if not IMAGE_DIR.exists():

    raise FileNotFoundError(
        f"No existe IMAGE_DIR: {IMAGE_DIR}"
    )


# ============================================================
# 4. LEER NOMBRES DE LAS IMÁGENES
# ============================================================

data = []

pattern = r'^(\d)__(.+?)__(-?\d+\.\d+)_(\d+\.\d+)\.png$'


for p in sorted(IMAGE_DIR.glob("*.png")):

    m = re.match(pattern, p.name)

    if m and int(m.group(1)) in (0, 1):

        data.append(
            {
                "filename": p.name,
                "filepath": str(p),
                "label": int(m.group(1))
            }
        )


df = pd.DataFrame(data)


if df.empty:

    raise RuntimeError(
        "No se encontraron imágenes con el formato esperado."
    )


print("\n============================================================")
print("INFORMACIÓN DEL DATASET")
print("============================================================")

print(f"Total de imágenes: {len(df)}")

print("\nCantidad por clase:")

print(
    df["label"]
    .value_counts()
    .sort_index()
    .rename(index={0: "No barco", 1: "Barco"})
)


# ============================================================
# 5. CARGAR IMÁGENES
# ============================================================

print("\nCargando imágenes...")

X = np.asarray(
    [
        img_to_array(
            load_img(
                p,
                color_mode="rgb",
                target_size=IMG_SIZE
            )
        )
        for p in df.filepath
    ],
    dtype="float32"
)


# Normalización 0-1
X = X / 255.0


y = df.label.to_numpy(dtype="int32")


print(f"Forma de X: {X.shape}")
print(f"Forma de y: {y.shape}")


# ============================================================
# 6. SEPARAR TEST FINAL
# ============================================================
#
# IMPORTANTE:
#
# El 10% de TEST queda completamente separado.
#
# No se utiliza:
#   - para entrenamiento
#   - para validación cruzada
#   - para buscar el umbral
#
# Solo se utiliza al final.
# ============================================================

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=SEED
)


print("\n============================================================")
print("DIVISIÓN DE DATOS")
print("============================================================")

print(f"Train + Validation: {len(X_trainval)}")
print(f"Test final:         {len(X_test)}")


# ============================================================
# 7. DATA AUGMENTATION
# ============================================================
#
# Estas transformaciones se ejecutan durante el entrenamiento.
#
# NO modifican permanentemente las imágenes originales.
# ============================================================

augmentation = Sequential(
    [

        RandomFlip(
            "horizontal"
        ),

        RandomRotation(
            0.05
        ),

        RandomZoom(
            height_factor=(-0.10, 0.10),
            width_factor=(-0.10, 0.10)
        ),

        RandomTranslation(
            height_factor=0.05,
            width_factor=0.05
        ),

        RandomContrast(
            0.10
        ),

        RandomBrightness(
            0.10
        )

    ],
    name="augmentation"
)


# ============================================================
# 8. CREAR MODELO V2
# ============================================================

def create_model_A_v2():

    model = Sequential(
        [

            Input(
                shape=(80, 80, 3)
            ),

            # ------------------------------------------------
            # Data augmentation
            # ------------------------------------------------

            augmentation,

            # ------------------------------------------------
            # BLOQUE 1
            # ------------------------------------------------

            Conv2D(
                32,
                3,
                padding="same",
                activation="relu"
            ),

            BatchNormalization(),

            MaxPooling2D(
                pool_size=2
            ),

            # ------------------------------------------------
            # BLOQUE 2
            # ------------------------------------------------

            Conv2D(
                64,
                3,
                padding="same",
                activation="relu"
            ),

            BatchNormalization(),

            MaxPooling2D(
                pool_size=2
            ),

            # ------------------------------------------------
            # BLOQUE 3
            # ------------------------------------------------

            Conv2D(
                128,
                3,
                padding="same",
                activation="relu"
            ),

            BatchNormalization(),

            SpatialDropout2D(
                0.15
            ),

            MaxPooling2D(
                pool_size=2
            ),

            # ------------------------------------------------
            # BLOQUE 4
            # ------------------------------------------------

            Conv2D(
                256,
                3,
                padding="same",
                activation="relu"
            ),

            BatchNormalization(),

            SpatialDropout2D(
                0.20
            ),

            MaxPooling2D(
                pool_size=2
            ),

            # ------------------------------------------------
            # REDUCCIÓN DE CARACTERÍSTICAS
            # ------------------------------------------------

            GlobalAveragePooling2D(),

            # ------------------------------------------------
            # CAPA DENSA
            # ------------------------------------------------

            Dense(
                128,
                activation="relu"
            ),

            BatchNormalization(),

            Dropout(
                0.40
            ),

            # ------------------------------------------------
            # SALIDA
            # ------------------------------------------------

            Dense(
                1,
                activation="sigmoid"
            )

        ],
        name="Modelo_Barcos_A_V2"
    )


    # --------------------------------------------------------
    # OPTIMIZADOR
    # --------------------------------------------------------

    optimizer = AdamW(
        learning_rate=LR,
        weight_decay=1e-4
    )


    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )


    return model


# ============================================================
# 9. CALLBACKS
# ============================================================

def create_callbacks():

    return [

        EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),

        ModelCheckpoint(
            filepath=str(BEST_MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )

    ]


# ============================================================
# 10. VALIDACIÓN CRUZADA 5-FOLD
# ============================================================
#
# El TEST sigue completamente separado.
# ============================================================

print("\n============================================================")
print("VALIDACIÓN CRUZADA 5-FOLD")
print("============================================================")


rows = []


skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED
)


for fold, (tr, va) in enumerate(
    skf.split(X_trainval, y_trainval),
    1
):

    print(f"\n---------------- FOLD {fold}/5 ----------------")


    model_cv = create_model_A_v2()


    model_cv.fit(
        X_trainval[tr],
        y_trainval[tr],

        validation_data=(
            X_trainval[va],
            y_trainval[va]
        ),

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        verbose=0,

        callbacks=create_callbacks()
    )


    # --------------------------------------------------------
    # PREDICCIONES
    # --------------------------------------------------------

    prob = model_cv.predict(
        X_trainval[va],
        verbose=0
    ).ravel()


    pred = (
        prob >= THRESHOLD_DEFAULT
    ).astype(int)


    # --------------------------------------------------------
    # MÉTRICAS
    # --------------------------------------------------------

    acc_fold = accuracy_score(
        y_trainval[va],
        pred
    )

    prec_fold = precision_score(
        y_trainval[va],
        pred,
        zero_division=0
    )

    rec_fold = recall_score(
        y_trainval[va],
        pred,
        zero_division=0
    )

    f1_fold = f1_score(
        y_trainval[va],
        pred,
        zero_division=0
    )


    rows.append(
        [
            fold,
            acc_fold,
            prec_fold,
            rec_fold,
            f1_fold
        ]
    )


    print(
        f"Accuracy : {acc_fold * 100:.2f}%"
    )

    print(
        f"Precision: {prec_fold * 100:.2f}%"
    )

    print(
        f"Recall   : {rec_fold * 100:.2f}%"
    )

    print(
        f"F1       : {f1_fold * 100:.2f}%"
    )


# ============================================================
# 11. GUARDAR RESULTADOS CV
# ============================================================

cv = pd.DataFrame(
    rows,
    columns=[
        "fold",
        "accuracy",
        "precision",
        "recall",
        "f1"
    ]
)


cv.to_csv(
    OUT / "resultados_validacion_cruzada_v2.csv",
    index=False
)


print("\n============================================================")
print("PROMEDIO VALIDACIÓN CRUZADA")
print("============================================================")


cv_mean = cv.mean(
    numeric_only=True
)


print(
    cv_mean * 100
)


# ============================================================
# 12. ENTRENAMIENTO FINAL
# ============================================================
#
# Aquí se utiliza nuevamente Train + Validation.
#
# El TEST sigue separado.
# ============================================================

print("\n============================================================")
print("ENTRENAMIENTO FINAL")
print("============================================================")


model = create_model_A_v2()


history = model.fit(
    X_trainval,
    y_trainval,

    validation_split=VAL_SIZE,

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    shuffle=True,

    verbose=1,

    callbacks=create_callbacks()
)


# ============================================================
# 13. GUARDAR MODELO
# ============================================================

model.save(
    MODEL_PATH
)


print(
    f"\nModelo V2 guardado en:\n"
    f"{MODEL_PATH.resolve()}"
)


# ============================================================
# 14. BUSCAR MEJOR UMBRAL
# ============================================================
#
# IMPORTANTE:
#
# Se crea un conjunto de validación independiente
# para seleccionar el umbral.
#
# El TEST NO participa.
# ============================================================

print("\n============================================================")
print("BÚSQUEDA DEL MEJOR UMBRAL")
print("============================================================")


# Separar una validación interna
X_train_inner, X_val_inner, y_train_inner, y_val_inner = train_test_split(
    X_trainval,
    y_trainval,
    test_size=VAL_SIZE,
    stratify=y_trainval,
    random_state=SEED
)


# Crear modelo temporal para obtener probabilidades
threshold_model = create_model_A_v2()


threshold_model.fit(
    X_train_inner,
    y_train_inner,

    validation_data=(
        X_val_inner,
        y_val_inner
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    verbose=0,

    callbacks=create_callbacks()
)


val_prob = threshold_model.predict(
    X_val_inner,
    verbose=0
).ravel()


best_threshold = THRESHOLD_DEFAULT

best_f1 = -1

threshold_rows = []


for threshold in np.arange(
    0.30,
    0.71,
    0.01
):

    val_pred = (
        val_prob >= threshold
    ).astype(int)


    threshold_acc = accuracy_score(
        y_val_inner,
        val_pred
    )

    threshold_prec = precision_score(
        y_val_inner,
        val_pred,
        zero_division=0
    )

    threshold_rec = recall_score(
        y_val_inner,
        val_pred,
        zero_division=0
    )

    threshold_f1 = f1_score(
        y_val_inner,
        val_pred,
        zero_division=0
    )


    threshold_rows.append(
        [
            threshold,
            threshold_acc,
            threshold_prec,
            threshold_rec,
            threshold_f1
        ]
    )


    if threshold_f1 > best_f1:

        best_f1 = threshold_f1

        best_threshold = threshold


threshold_df = pd.DataFrame(
    threshold_rows,
    columns=[
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1"
    ]
)


threshold_df.to_csv(
    OUT / "busqueda_umbral.csv",
    index=False
)


print(
    f"Umbral estándar: {THRESHOLD_DEFAULT:.2f}"
)

print(
    f"Mejor umbral encontrado: {best_threshold:.2f}"
)

print(
    f"Mejor F1 en validación: {best_f1 * 100:.2f}%"
)


# ============================================================
# 15. TEST FINAL
# ============================================================
#
# ESTE ES EL RESULTADO MÁS IMPORTANTE.
#
# El conjunto TEST no se utilizó para:
#
#   - entrenamiento
#   - selección del umbral
#   - validación cruzada
#
# ============================================================

print("\n============================================================")
print("TEST FINAL")
print("============================================================")


prob = model.predict(
    X_test,
    batch_size=BATCH_SIZE,
    verbose=0
).ravel()


# ------------------------------------------------------------
# Resultado con umbral 0.50
# ------------------------------------------------------------

pred_default = (
    prob >= THRESHOLD_DEFAULT
).astype(int)


acc_default = accuracy_score(
    y_test,
    pred_default
)

prec_default = precision_score(
    y_test,
    pred_default,
    zero_division=0
)

rec_default = recall_score(
    y_test,
    pred_default,
    zero_division=0
)

f1_default = f1_score(
    y_test,
    pred_default,
    zero_division=0
)

cm_default = confusion_matrix(
    y_test,
    pred_default,
    labels=[0, 1]
)


# ------------------------------------------------------------
# Resultado con mejor umbral
# ------------------------------------------------------------

pred_best = (
    prob >= best_threshold
).astype(int)


acc_best = accuracy_score(
    y_test,
    pred_best
)

prec_best = precision_score(
    y_test,
    pred_best,
    zero_division=0
)

rec_best = recall_score(
    y_test,
    pred_best,
    zero_division=0
)

f1_best = f1_score(
    y_test,
    pred_best,
    zero_division=0
)

cm_best = confusion_matrix(
    y_test,
    pred_best,
    labels=[0, 1]
)


# ============================================================
# 16. MOSTRAR RESULTADOS
# ============================================================

print("\n============================================================")
print("RESULTADOS CON UMBRAL 0.50")
print("============================================================")

print(
    f"Accuracy : {acc_default * 100:.2f}%"
)

print(
    f"Precision: {prec_default * 100:.2f}%"
)

print(
    f"Recall   : {rec_default * 100:.2f}%"
)

print(
    f"F1       : {f1_default * 100:.2f}%"
)

print(
    "\nMatriz de confusión:"
)

print(cm_default)


print(
    "\n============================================================"
)

print(
    f"RESULTADOS CON UMBRAL {best_threshold:.2f}"
)

print(
    "============================================================"
)

print(
    f"Accuracy : {acc_best * 100:.2f}%"
)

print(
    f"Precision: {prec_best * 100:.2f}%"
)

print(
    f"Recall   : {rec_best * 100:.2f}%"
)

print(
    f"F1       : {f1_best * 100:.2f}%"
)

print(
    "\nMatriz de confusión:"
)

print(cm_best)


# ============================================================
# 17. REPORTE DE CLASIFICACIÓN
# ============================================================

print("\n============================================================")
print("CLASSIFICATION REPORT")
print("============================================================")


print(
    classification_report(
        y_test,
        pred_best,
        target_names=[
            "No barco",
            "Barco"
        ],
        digits=4,
        zero_division=0
    )
)


# ============================================================
# 18. GUARDAR MATRIZ DE CONFUSIÓN
# ============================================================

np.savetxt(
    OUT / "matriz_confusion_v2.csv",
    cm_best,
    delimiter=",",
    fmt="%d"
)


# ============================================================
# 19. GUARDAR MÉTRICAS
# ============================================================

metrics = {

    "modelo": "Barcos Modelo A V2",

    "accuracy_threshold_050": float(
        acc_default
    ),

    "precision_threshold_050": float(
        prec_default
    ),

    "recall_threshold_050": float(
        rec_default
    ),

    "f1_threshold_050": float(
        f1_default
    ),

    "best_threshold": float(
        best_threshold
    ),

    "accuracy_best_threshold": float(
        acc_best
    ),

    "precision_best_threshold": float(
        prec_best
    ),

    "recall_best_threshold": float(
        rec_best
    ),

    "f1_best_threshold": float(
        f1_best
    ),

    "test_samples": int(
        len(y_test)
    ),

    "trainval_samples": int(
        len(y_trainval)
    ),

    "seed": SEED,

    "image_size": list(
        IMG_SIZE
    ),

    "batch_size": BATCH_SIZE,

    "epochs_max": EPOCHS,

    "learning_rate": LR

}


with open(
    OUT / "metricas_test_v2.json",
    "w",
    encoding="utf8"
) as f:

    json.dump(
        metrics,
        f,
        indent=2
    )


# ============================================================
# 20. GRÁFICA DE ACCURACY
# ============================================================

plt.figure()

plt.plot(
    history.history["accuracy"],
    label="Train"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation"
)

plt.xlabel(
    "Época"
)

plt.ylabel(
    "Accuracy"
)

plt.title(
    "Accuracy - Modelo Barcos A V2"
)

plt.legend()

plt.grid()

plt.tight_layout()

plt.savefig(
    OUT / "accuracy_entrenamiento_v2.png",
    dpi=150
)

plt.close()


# ============================================================
# 21. GRÁFICA DE LOSS
# ============================================================

plt.figure()

plt.plot(
    history.history["loss"],
    label="Train"
)

plt.plot(
    history.history["val_loss"],
    label="Validation"
)

plt.xlabel(
    "Época"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "Loss - Modelo Barcos A V2"
)

plt.legend()

plt.grid()

plt.tight_layout()

plt.savefig(
    OUT / "loss_entrenamiento_v2.png",
    dpi=150
)

plt.close()


# ============================================================
# 22. GRÁFICA MATRIZ DE CONFUSIÓN
# ============================================================

plt.figure()

plt.imshow(
    cm_best
)

plt.title(
    f"Matriz de confusión - Test V2"
)

plt.xlabel(
    "Predicción"
)

plt.ylabel(
    "Real"
)

plt.xticks(
    [0, 1],
    [
        "No barco",
        "Barco"
    ]
)

plt.yticks(
    [0, 1],
    [
        "No barco",
        "Barco"
    ]
)


for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            str(cm_best[i, j]),
            ha="center",
            va="center"
        )


plt.tight_layout()

plt.savefig(
    OUT / "matriz_confusion_v2.png",
    dpi=150
)

plt.close()


# ============================================================
# 23. GRÁFICA DEL UMBRAL
# ============================================================

plt.figure()

plt.plot(
    threshold_df["threshold"],
    threshold_df["accuracy"],
    label="Accuracy"
)

plt.plot(
    threshold_df["threshold"],
    threshold_df["precision"],
    label="Precision"
)

plt.plot(
    threshold_df["threshold"],
    threshold_df["recall"],
    label="Recall"
)

plt.plot(
    threshold_df["threshold"],
    threshold_df["f1"],
    label="F1"
)

plt.axvline(
    best_threshold,
    linestyle="--",
    label=f"Mejor umbral = {best_threshold:.2f}"
)

plt.xlabel(
    "Umbral"
)

plt.ylabel(
    "Métrica"
)

plt.title(
    "Métricas según el umbral"
)

plt.legend()

plt.grid()

plt.tight_layout()

plt.savefig(
    OUT / "analisis_umbral_v2.png",
    dpi=150
)

plt.close()


# ============================================================
# 24. RESUMEN FINAL
# ============================================================

print("\n============================================================")
print("RESUMEN FINAL MODELO A V2")
print("============================================================")

print(
    f"Accuracy (0.50) : {acc_default * 100:.2f}%"
)

print(
    f"Precision (0.50): {prec_default * 100:.2f}%"
)

print(
    f"Recall (0.50)   : {rec_default * 100:.2f}%"
)

print(
    f"F1 (0.50)       : {f1_default * 100:.2f}%"
)

print(
    "\n--------------------------------------------"
)

print(
    f"Mejor umbral    : {best_threshold:.2f}"
)

print(
    f"Accuracy final  : {acc_best * 100:.2f}%"
)

print(
    f"Precision final : {prec_best * 100:.2f}%"
)

print(
    f"Recall final    : {rec_best * 100:.2f}%"
)

print(
    f"F1 final        : {f1_best * 100:.2f}%"
)

print(
    "\n--------------------------------------------"
)

print(
    f"Meta 98%: "
    f"{'CUMPLIDA' if acc_best >= 0.98 else 'NO CUMPLIDA'}"
)

print(
    "\nModelo guardado:"
)

print(
    MODEL_PATH.resolve()
)

print(
    "\nMejor modelo durante entrenamiento:"
)

print(
    BEST_MODEL_PATH.resolve()
)

print(
    "\nProceso finalizado correctamente."
)

print(
    "============================================================"
)