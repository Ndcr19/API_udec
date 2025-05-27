import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import pickle
import json
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

class IndoorLocalizationModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder_etiqueta = LabelEncoder()
        self.label_encoder_ubicacion = LabelEncoder()
        self.knn_model = None
        self.rf_model = None
        self.feature_columns = ['COR_X', 'COR_Y', 'ALTITUD', 'INTENSIDAD_SEÑAL']
        self.bssid_features = {}
        self.trained = False
        
    def load_data_from_string(self, data_string):
    
        """Carga los datos desde el string proporcionado"""
    
        lines = data_string.strip().split('\n')
        data = []

        for line in lines:
            if line.strip():
                parts = line.split(',')
                if len(parts) >= 7:
                    try:
                        row = {
                            'COR_X': float(parts[0].replace('"', '')),
                            'COR_Y': float(parts[1].replace('"', '')),
                            'ALTITUD': float(parts[2].replace('"', '')),
                            'BSSID': parts[3].strip().replace('"', ''),
                            'INTENSIDAD_SEÑAL': float(parts[4].replace('"', '')),
                            'ETIQUETA': parts[5].strip().replace('"', ''),
                            'UBICACION': parts[6].strip().replace('"', '')
                        }
                        data.append(row)
                    except ValueError as e:
                        print(f"Error en línea: {line} → {e}")
        
        return pd.DataFrame(data)

    
    def preprocess_data(self, df):
        """Preprocesa los datos creando características de BSSID"""
        # Obtener todos los BSSID únicos
        unique_bssids = df['BSSID'].unique()
        
        # Crear un diccionario para mapear cada punto a sus mediciones
        location_data = defaultdict(list)
        
        for _, row in df.iterrows():
            key = (row['COR_X'], row['COR_Y'], row['ALTITUD'], row['ETIQUETA'], row['UBICACION'])
            location_data[key].append({
                'BSSID': row['BSSID'],
                'INTENSIDAD_SEÑAL': row['INTENSIDAD_SEÑAL']
            })
        
        # Crear el dataset final con características agregadas
        processed_data = []
        
        for location_key, measurements in location_data.items():
            cor_x, cor_y, altitud, etiqueta, ubicacion = location_key
            
            # Crear vector de características para BSSIDs
            bssid_features = {}
            for bssid in unique_bssids:
                # Encontrar todas las mediciones para este BSSID en esta ubicación
                bssid_measurements = [m['INTENSIDAD_SEÑAL'] for m in measurements if m['BSSID'] == bssid]
                
                if bssid_measurements:
                    # Usar la mejor señal (menos negativa) como característica
                    bssid_features[f'BSSID_{bssid}'] = max(bssid_measurements)
                else:
                    # Si no hay medición para este BSSID, usar un valor muy bajo
                    bssid_features[f'BSSID_{bssid}'] = -100
            
            # Agregar características adicionales
            intensidades = [m['INTENSIDAD_SEÑAL'] for m in measurements]
            
            row_data = {
                'COR_X': cor_x,
                'COR_Y': cor_y,
                'ALTITUD': altitud,
                'ETIQUETA': etiqueta,
                'UBICACION': ubicacion,
                'INTENSIDAD_PROMEDIO': np.mean(intensidades),
                'INTENSIDAD_MAX': max(intensidades),
                'INTENSIDAD_MIN': min(intensidades),
                'NUM_BSSIDS': len(set(m['BSSID'] for m in measurements)),
                **bssid_features
            }
            
            processed_data.append(row_data)
        
        return pd.DataFrame(processed_data), unique_bssids
    
    def train(self, data_string):
        """Entrena el modelo con los datos proporcionados"""
        print("Cargando datos...")
        df = self.load_data_from_string(data_string)
        print(f"Datos cargados: {len(df)} registros")
        
        # Preprocesar datos
        print("Preprocesando datos...")
        processed_df, unique_bssids = self.preprocess_data(df)
        self.unique_bssids = unique_bssids
        print(f"Datos procesados: {len(processed_df)} ubicaciones únicas")
        
        # Separar características y etiquetas
        feature_cols = ['COR_X', 'COR_Y', 'ALTITUD', 'INTENSIDAD_PROMEDIO', 'INTENSIDAD_MAX', 
                       'INTENSIDAD_MIN', 'NUM_BSSIDS'] + [f'BSSID_{bssid}' for bssid in unique_bssids]
        
        # Filtrar etiquetas y ubicaciones con al menos 2 muestras
        et_counts = processed_df['ETIQUETA'].value_counts()
        ub_counts = processed_df['UBICACION'].value_counts()

        processed_df = processed_df[
            processed_df['ETIQUETA'].isin(et_counts[et_counts >= 2].index) &
            processed_df['UBICACION'].isin(ub_counts[ub_counts >= 2].index)
        ]

        
        X = processed_df[feature_cols]
        y_etiqueta = processed_df['ETIQUETA']
        y_ubicacion = processed_df['UBICACION']
        
        # Codificar etiquetas
        y_etiqueta_encoded = self.label_encoder_etiqueta.fit_transform(y_etiqueta)
        y_ubicacion_encoded = self.label_encoder_ubicacion.fit_transform(y_ubicacion)
        
        # Escalar características
        X_scaled = self.scaler.fit_transform(X)
        
        # Dividir datos para entrenamiento y prueba
        X_train = X_scaled
        y_train_et = y_etiqueta_encoded

        
        # Entrenar modelos
        print("Entrenando modelo KNN...")
        self.knn_model = KNeighborsClassifier(n_neighbors=3, weights='distance', metric='minkowski')
        self.knn_model.fit(X_train, y_train_et)
        
        print("Entrenando modelo Random Forest...")
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.rf_model.fit(X_train, y_train_et)
        
        
        # Guardar datos de referencia para similitud
        self.reference_data = processed_df
        self.feature_columns = feature_cols
        self.trained = True
        
        print("¡Entrenamiento completado!")
        return processed_df
    
    def create_fingerprint_from_measurements(self, measurements):
        """Crea un fingerprint a partir de mediciones de WiFi"""
        fingerprint = {}
        
        # Inicializar todos los BSSIDs con valor muy bajo
        for bssid in self.unique_bssids:
            fingerprint[f'BSSID_{bssid}'] = -100
        
        # Procesar mediciones recibidas
        bssid_measurements = defaultdict(list)
        for measurement in measurements:
            bssid = measurement['BSSID']
            intensity = measurement['INTENSIDAD_SEÑAL']
            bssid_measurements[bssid].append(intensity)
        
        # Actualizar fingerprint con las mejores señales
        for bssid, intensities in bssid_measurements.items():
            if f'BSSID_{bssid}' in fingerprint:
                fingerprint[f'BSSID_{bssid}'] = max(intensities)
        
        # Calcular estadísticas adicionales
        all_intensities = [m['INTENSIDAD_SEÑAL'] for m in measurements]
        fingerprint.update({
            'INTENSIDAD_PROMEDIO': np.mean(all_intensities),
            'INTENSIDAD_MAX': max(all_intensities),
            'INTENSIDAD_MIN': min(all_intensities),
            'NUM_BSSIDS': len(bssid_measurements)
        })
        
        return fingerprint
    
    def predict_location(self, measurements, coordenadas_aprox=None, top_k=3):
        """Predice la ubicación basada en mediciones WiFi"""
        if not self.trained:
            raise Exception("El modelo no ha sido entrenado")
        
        # Crear fingerprint de la consulta
        query_fingerprint = self.create_fingerprint_from_measurements(measurements)
        
        # Si se proporcionan coordenadas aproximadas, agregarlas
        if coordenadas_aprox:
            query_fingerprint.update({
                'COR_X': coordenadas_aprox.get('COR_X', 0),
                'COR_Y': coordenadas_aprox.get('COR_Y', 0),
                'ALTITUD': coordenadas_aprox.get('ALTITUD', 0)
            })
        else:
            query_fingerprint.update({'COR_X': 0, 'COR_Y': 0, 'ALTITUD': 0})
        
        # Preparar datos para predicción
        query_df = pd.DataFrame([query_fingerprint])
        query_features = query_df[self.feature_columns]
        query_scaled = self.scaler.transform(query_features)
        
        # Predicción con KNN
        knn_pred = self.knn_model.predict(query_scaled)[0]
        knn_proba = self.knn_model.predict_proba(query_scaled)[0]
        
        # Predicción con Random Forest
        rf_pred = self.rf_model.predict(query_scaled)[0]
        rf_proba = self.rf_model.predict_proba(query_scaled)[0]
        
        # Calcular similitudes con todas las ubicaciones de referencia
        similarities = []
        reference_features = self.reference_data[self.feature_columns]
        reference_scaled = self.scaler.transform(reference_features)
        
        # Similitud coseno
        cos_similarities = cosine_similarity(query_scaled, reference_scaled)[0]
        
        # Distancia euclidiana (convertida a similitud)
        eucl_distances = euclidean_distances(query_scaled, reference_scaled)[0]
        eucl_similarities = 1 / (1 + eucl_distances)
        
        for idx, (_, row) in enumerate(self.reference_data.iterrows()):
            similarity_score = (cos_similarities[idx] + eucl_similarities[idx]) / 2
            similarities.append({
                'etiqueta': row['ETIQUETA'],
                'ubicacion': row['UBICACION'],
                'coordenadas': {
                    'COR_X': row['COR_X'],
                    'COR_Y': row['COR_Y'],
                    'ALTITUD': row['ALTITUD']
                },
                'similitud_coseno': cos_similarities[idx],
                'similitud_euclidiana': eucl_similarities[idx],
                'similitud_promedio': similarity_score
            })
        
        # Ordenar por similitud
        similarities.sort(key=lambda x: x['similitud_promedio'], reverse=True)
        
        # Resultados
        result = {
            'prediccion_knn': {
                'etiqueta': self.label_encoder_etiqueta.inverse_transform([knn_pred])[0],
                'confianza': max(knn_proba)
            },
            'prediccion_rf': {
                'etiqueta': self.label_encoder_etiqueta.inverse_transform([rf_pred])[0],
                'confianza': max(rf_proba)
            },
            'top_similares': similarities[:top_k],
            'ubicacion_mas_probable': similarities[0]
        }
        
        return result
    
    def save_model(self, filepath):
        """Guarda el modelo entrenado"""
        model_data = {
            'scaler': self.scaler,
            'label_encoder_etiqueta': self.label_encoder_etiqueta,
            'label_encoder_ubicacion': self.label_encoder_ubicacion,
            'knn_model': self.knn_model,
            'rf_model': self.rf_model,
            'reference_data': self.reference_data,
            'feature_columns': self.feature_columns,
            'unique_bssids': self.unique_bssids,
            'trained': self.trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Modelo guardado en: {filepath}")
    
    def load_model(self, filepath):
        """Carga un modelo previamente entrenado"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.scaler = model_data['scaler']
        self.label_encoder_etiqueta = model_data['label_encoder_etiqueta']
        self.label_encoder_ubicacion = model_data['label_encoder_ubicacion']
        self.knn_model = model_data['knn_model']
        self.rf_model = model_data['rf_model']
        self.reference_data = model_data['reference_data']
        self.feature_columns = model_data['feature_columns']
        self.unique_bssids = model_data['unique_bssids']
        self.trained = model_data['trained']
        
        print(f"Modelo cargado desde: {filepath}")

# Ejemplo de uso
if __name__ == "__main__":
    # Datos de ejemplo (usar los datos reales proporcionados)
    # Cargar tu dataset real desde CSV
    df = pd.read_csv("datos_wifi_unidos.csv", encoding="latin1", quotechar='"')
    df.rename(columns={"INTENSIDAD_SEÃ‘AL": "INTENSIDAD_SEÑAL"}, inplace=True)
    print(df.columns.tolist())


    # Convertir el DataFrame a la lista de líneas que espera tu método `train()`
    data_lines = df.apply(lambda row: ",".join(map(str, row)), axis=1).tolist()
    data_string_real = "\n".join(data_lines)

    
    # Crear y entrenar el modelo
    model = IndoorLocalizationModel()

    # Entrenar el modelo con datos reales
    processed_data = model.train(data_string_real)
    
    # Ejemplo de predicción
    # Simular mediciones desde una app móvil
    mediciones_ejemplo = [
        {'BSSID': 'dc:ae:eb:e4:ce:8c', 'INTENSIDAD_SEÑAL': -37},
        {'BSSID': '34:20:e3:d2:2d:2c', 'INTENSIDAD_SEÑAL': -63},
        {'BSSID': '34:20:e3:d2:2d:28', 'INTENSIDAD_SEÑAL': -55}
    ]
    
    # Predecir ubicación
    resultado = model.predict_location(mediciones_ejemplo)
    
    print("\n=== RESULTADO DE PREDICCIÓN ===")
    print(f"Predicción KNN: {resultado['prediccion_knn']['etiqueta']} (confianza: {resultado['prediccion_knn']['confianza']:.3f})")
    print(f"Predicción RF: {resultado['prediccion_rf']['etiqueta']} (confianza: {resultado['prediccion_rf']['confianza']:.3f})")
    print(f"\nUbicación más probable: {resultado['ubicacion_mas_probable']['etiqueta']}")
    print(f"Similitud: {resultado['ubicacion_mas_probable']['similitud_promedio']:.3f}")
    print(f"Coordenadas: {resultado['ubicacion_mas_probable']['coordenadas']}")
    
    print(f"\nTop 3 ubicaciones similares:")
    for i, loc in enumerate(resultado['top_similares'][:3], 1):
        print(f"{i}. {loc['etiqueta']} - Similitud: {loc['similitud_promedio']:.3f}")
    
    # Guardar el modelo
    model.save_model("modelo_localizacion_indoor.pkl")