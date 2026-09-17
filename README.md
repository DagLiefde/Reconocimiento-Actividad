# Reconocimiento de Actividades Humanas a partir de Sensores Inerciales de Smartphones (HAR)

Proyecto de Machine Learning — Entrega 1 (propuesta y línea base).

Reporte completo de la propuesta: [`docs/propuesta_entrega1.pdf`](docs/propuesta_entrega1.pdf)

## 1. Contexto de aplicación

El reconocimiento de actividad humana (*Human Activity Recognition*, HAR) a partir de sensores inerciales embebidos en smartphones es un problema central en *ubiquitous computing*, salud digital y bienestar. A diferencia de sistemas basados en cámaras, un smartphone portado por el usuario permite monitoreo continuo, de bajo costo y no intrusivo.

Aplicaciones prácticas: apps de *fitness* que cuantifican actividad física diaria, sistemas de asistencia a adultos mayores (detección de caídas / inactividad prolongada), interfaces adaptativas según contexto físico del usuario, y telemetría de salud en poblaciones clínicas (p. ej. detección de congelamiento de la marcha en pacientes con Parkinson).

## 2. Objetivo de machine learning

**Predecir la actividad física que realiza una persona (6 clases) a partir de señales inerciales de un smartphone (acelerómetro y giroscopio de 3 ejes) capturadas en ventanas de 2.56 s.**

Es un problema de **clasificación multiclase supervisada**: dado un vector de entrada `x` (señal cruda de 9 canales × 128 pasos, o el vector de 561 features ya extraídas), aprender `f(x) → y`, con:

```
y ∈ {WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING}
```

## 3. Dataset

**[UCI HAR Dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)** — 30 voluntarios (19-48 años), Samsung Galaxy S II en la cintura, acelerómetro + giroscopio a 50 Hz, ventanas de 128 muestras (2.56 s, 50% traslape).

| Partición | # Sujetos | # Ventanas | # Features |
|---|---|---|---|
| Train | 21 | 7,352 | 561 |
| Test | 9 | 2,947 | 561 |
| **Total** | **30** | **10,299** | **561** |

- **Tamaño en disco:** ~270 MB descomprimido / ~61 MB comprimido (`.zip`).
- **Tipo de datos:** señales inerciales crudas (9 canales × 128 pasos por ventana) **y** un vector de 561 características de dominio tiempo/frecuencia ya normalizadas en `[-1, 1]`.
- **Distribución de clases:** razonablemente balanceada (ratio máx/mín ≈ 1.43); ver detalle en el reporte.
- **Partición:** *subject-wise* (70%/30% de sujetos, no de ventanas), evitando fuga de información entre train y test.

El dataset **no se versiona directamente en este repositorio** (60 MB, y su licencia original prohíbe uso comercial). En su lugar, ver [`data/README.md`](data/README.md) para el enlace oficial de descarga y el script de una línea que lo coloca en `data/raw/UCI_HAR_Dataset.zip`, desde donde `src/data_loader.py` lo descomprime automáticamente.

## 4. Métricas de desempeño

**Machine learning:** accuracy global, precisión/recall/F1 por clase (macro y ponderado), matriz de confusión, y validación cruzada por sujeto (*leave-subjects-out*) para estimar generalización a usuarios nuevos.

**Negocio:** latencia de inferencia por ventana (ms) para viabilidad de *on-device inference*, consumo computacional/energético, tasa de falsos negativos en clases de riesgo (p. ej. detección de inactividad/caídas), y robustez ante sujetos/dispositivos no vistos en entrenamiento.

## 5. Referencias y resultados previos

1. Anguita, D. et al. (2013). *A Public Domain Dataset for Human Activity Recognition Using Smartphones*. ESANN 2013. — SVM: **96.0%** accuracy.
2. Murad, A.; Pyun, J.-Y. (2017). *Deep Recurrent Neural Networks for Human Activity Recognition*. **Sensors**, 17(11), 2556. [doi:10.3390/s17112556](https://doi.org/10.3390/s17112556). — Referencia principal; DRNN unidireccional (4 capas): **96.7%** accuracy, 96.8% precisión, 96.7% recall, F1 = 0.96.
3. Jiang, W.; Yin, Z. (2015). CNN baseline usada en (2): **95.2%** accuracy.
4. Chandan Kumar, R. et al. (2016). ELM secuencial baseline usada en (2): **93.3%** accuracy.

Estos valores fijan el umbral de desempeño esperado: superar la línea base de ELM (93.3%) y acercarse/igualar el 96-97% reportado por SVM, CNN y DRNN.

---

## Estructura del repositorio

```
.
├── data/
│   ├── README.md                   # instrucciones para descargar el dataset
│   └── raw/                        # (vacío en git) aquí se coloca UCI_HAR_Dataset.zip
├── docs/
│   └── propuesta_entrega1.pdf      # reporte académico de la Entrega 1
├── src/
│   ├── data_loader.py              # carga de features y señales crudas
│   └── train_baseline.py           # pipeline base: entrena + evalúa + guarda métricas
├── results/                        # métricas generadas por los scripts (no versionadas)
├── requirements.txt
└── README.md
```

## Cómo ejecutar el pipeline base

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Descarga el dataset (ver data/README.md para más detalle)
mkdir -p data/raw
curl -L -o data/raw/UCI_HAR_Dataset.zip \
  "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"

# Entrena y evalúa una línea base (Random Forest o SVM) sobre las 561 features.
# La primera ejecución descomprime automáticamente data/raw/UCI_HAR_Dataset.zip
python src/train_baseline.py --model random_forest
python src/train_baseline.py --model svm
```

Cada corrida imprime accuracy, F1 (macro/ponderado), matriz de confusión y latencia de inferencia por ventana, y guarda un JSON con las métricas completas en `results/`.

### Resultado de referencia (línea base incluida en este repo)

| Modelo | Accuracy | F1 macro |
|---|---|---|
| Random Forest (561 features) | ~92.9% | ~92.7% |

Sirve como punto de partida: el objetivo del proyecto es acercarse al 96-97% reportado en la literatura (Tabla de referencias, sección 5) mediante mejores modelos (SVM afinado, o arquitecturas de deep learning como las DRNN de Murad & Pyun, 2017, que trabajan directamente sobre las señales crudas en `src/data_loader.py::load_raw_signals`).

## Próximos pasos

- [ ] Ajuste de hiperparámetros de la línea base (SVM / Random Forest).
- [ ] Implementación de un modelo de deep learning (LSTM/DRNN) sobre las señales crudas.
- [ ] Validación *leave-subjects-out* para estimar generalización real.
- [ ] Análisis de latencia/tamaño de modelo para viabilidad de despliegue on-device.
