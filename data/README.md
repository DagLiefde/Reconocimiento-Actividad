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
