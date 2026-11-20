# EEG Pain Research (`eeg_pain_research`)

Pipeline en Python para el análisis espectral (Potencia Absoluta, Potencia Relativa y ERD/ERS%) y clasificación de intensidad de dolor evocado por láser en registros EEG de 62 canales (51 sujetos).

## Metodología implementada
- **Decodificación MCOS:** Lectura directa de etiquetas `Eventos1` desde archivos `.mat` v7.3 sin dependencias de MATLAB.
- **Control de fuga de datos:** Separación estricta de línea base pre-estímulo (`[-1s, 0s]`) y respuesta post-estímulo (`[0s, +1s]`) con Welch PSD, más escalado `StandardScaler` dentro de cada pliegue de validación.
- **Validación:** Intra-sujeto (`StratifiedKFold` 5-fold) e inter-sujeto (`Leave-One-Subject-Out` y `Hold-Out 70/30` agrupado por participante), comparando 62 canales completos vs. 20 canales somatosensoriales.

## Ejecución
1. Instalar dependencias: `pip install -r requirements.txt`
2. Verificar que `DIRECTORIO` en `eeg_pain_research.py` apunte a `C:\Datos_Control\`.
3. Ejecutar: `python eeg_pain_research.py`