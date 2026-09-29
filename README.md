
# Proyecto 2 - Modelo A

## Instalación
```bash
pip install tensorflow numpy pandas scikit-learn matplotlib pillow
```

## Orden de ejecución

1. Editar `IMAGE_DIR` en `entrenamiento_modelo_A.py`.
2. Ejecutar:
```bash
python entrenamiento_modelo_A.py
```
3. El modelo queda en:
`resultados_modelo_A/modelo_barcos_A.keras`
4. Ejecutar la UI:
```bash
python ui_modelo_A.py
```

## Evaluación en vivo

Para calcular automáticamente accuracy/precision/recall, la UI necesita conocer la etiqueta real. En una carpeta de evaluación etiquetada, los nombres deben empezar por:
- `1__` = barco
- `0__` = no barco

Si el profesor entrega un test verdaderamente ciego sin etiquetas, la UI puede hacer inferencia, pero no puede calcular accuracy real hasta conocer las etiquetas verdaderas. No se deben usar esas etiquetas durante el entrenamiento.

## E1
UI funcional: carpeta, imagen, inferencia y evaluación.

## E2
CNN, normalización, data augmentation, Batch Normalization, Dropout, Adam, learning rate, EarlyStopping y ReduceLROnPlateau.

## E3
Validación cruzada 5-fold y evaluación separada del test.

## E4
Accuracy, precision, recall y matriz de confusión en vivo.

## Nota
El código no garantiza 98% de accuracy. Si el resultado no alcanza 98%, se deben realizar experimentos controlados de optimización sin tocar el test ciego.
