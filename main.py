from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
from ModeloPrediccion import IndoorLocalizationModel  # Asegúrate de que este sea el nombre de tu clase y archivo

# Cargar el modelo previamente entrenado
modelo = IndoorLocalizationModel()
modelo.load_model("modelo_localizacion_indoor.pkl")

app = FastAPI()

# Definir el esquema de datos que recibirá la API
class Medicion(BaseModel):
    BSSID: str
    INTENSIDAD_SEÑAL: int

class RequestMediciones(BaseModel):
    mediciones: List[Medicion]

@app.post("/predecir_ubicacion/")
def predecir_ubicacion(data: RequestMediciones):
    try:
        # Convertir las mediciones al formato esperado por el modelo
        mediciones_formateadas = [
            {'BSSID': m.BSSID, 'INTENSIDAD_SEÑAL': m.INTENSIDAD_SEÑAL}
            for m in data.mediciones
        ]

        resultado = modelo.predict_location(mediciones_formateadas)

        return {
            "ubicacion_mas_probable": resultado['ubicacion_mas_probable'],
            "prediccion_knn": resultado['prediccion_knn'],
            "prediccion_rf": resultado['prediccion_rf'],
            "top_similares": resultado['top_similares'][:3],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# OPCIONAL: ejecuta directamente con `python api_localizacion.py`
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_localizacion:app", host="127.0.0.1", port=8000, reload=True)
