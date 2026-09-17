# Dataset

Este proyecto usa el **UCI HAR Dataset** (Human Activity Recognition Using
Smartphones), disponible públicamente en el repositorio de UC Irvine:

- Página oficial: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
- Descarga directa (zip, ~61 MB): https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip

## Cómo obtenerlo

```bash
mkdir -p data/raw
curl -L -o data/raw/UCI_HAR_Dataset.zip \
  "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
```

Una vez descargado en `data/raw/UCI_HAR_Dataset.zip`, los scripts en `src/`
lo descomprimen automáticamente en la primera ejecución (ver
`src/data_loader.py::ensure_extracted`).

El dataset no se versiona directamente en este repositorio (60 MB) para
mantenerlo liviano; además, su licencia original prohíbe el uso comercial
(ver `LICENSE-DATASET.txt`) y por buenas prácticas de MLOps los datos
crudos no deben vivir en control de versiones de código.

**Cita requerida** si usas este dataset en una publicación:

> Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L.
> Reyes-Ortiz. A Public Domain Dataset for Human Activity Recognition Using
> Smartphones. 21th European Symposium on Artificial Neural Networks,
> Computational Intelligence and Machine Learning, ESANN 2013. Bruges,
> Belgium 24-26 April 2013.
