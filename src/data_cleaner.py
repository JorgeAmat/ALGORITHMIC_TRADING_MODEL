import yfinance as yf
import pandas as pd
import numpy as np
import os
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import pandas as pd
from datetime import datetime
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


API_KEY = "PKCB2FGQTEQAIQXV5CXL6RJU4D" 
SECRET_KEY = "8gsG7rUpBvUaxAzw7HPw6EEypH42UpkR4VRYHaNJoYYU"
#Con esta clase definimos el objeto de configuración
class DataCleanerConfig:
    def __init__(self, source="alpaca", symbol="QQQ", interval="15m", start_date="2025-10-01", end_date="2025-10-22", csv_path=None):
        self.source = source
        self.symbol = symbol
        self.interval = interval
        self.start_date = start_date
        self.end_date = end_date
        self.csv_path = csv_path
      
        # Claves Alpaca
        self.api_key = API_KEY
        self.secret_key = SECRET_KEY
#DEFINIMOS EL CONSTRUCTOR
class DataCleaner:
    def __init__(self, cfg):
            self.cfg = cfg #Esta es la configuración, indica origen, fechas, rutas del csv...
            self.df = None

    # Cargamos el CSV    
        
    def _load_csv(self, path):
            if not os.path.exists(path):
                raise FileNotFoundError(f"❌ CSV no encontrado en: {path}")

            df = pd.read_csv(path)

            # Normalizamos columnas a minúsculas
            df.columns = [c.lower() for c in df.columns]

            # Aseguramos que datetime existe por si  acaso

            if "datetime" not in df.columns:
                raise ValueError("❌ El CSV debe contener una columna 'datetime'.")

            return df

    def _fetch_yfinance(self):

        
        #Descarga datos usando yfinance según la configuración dada en DataCleanerConfig
        import warnings
        warnings.filterwarnings("ignore")  # silencia los putos warnings

        data = yf.download(
            tickers=self.cfg.symbol,
            start=self.cfg.start_date,
            end=self.cfg.end_date,
            interval=self.cfg.interval,
            progress=False
        )

        if data is None or len(data) == 0:
            raise ValueError("yfinance no devolvió datos (posible problema de intervalo o fechas).")
        

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.columns.name = None

        #peequeño test
        # print("✅ Data descargada:", len(data))
        # print(data)
        # Renombramos las columnas
        data = data.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        })

        #Aseguramos que la columna se llame datetime, para que no haaya confusiones
        data = data.reset_index()
        if "Date" in data.columns:
            data = data.rename(columns={"Date": "datetime"})
        elif "Datetime" in data.columns:
            data = data.rename(columns={"Datetime": "datetime"})
        else:
            data["datetime"] = data.index
        data.columns = [c.lower() for c in data.columns]
        return data         

    def _fetch_alpaca(self):
        """Descarga datos usando Alpaca según config."""
        if not self.cfg.api_key or not self.cfg.secret_key:
            raise ValueError("Debes pasar api_key y secret_key en DataCleanerConfig para usar Alpaca.")

        client = StockHistoricalDataClient(api_key=self.cfg.api_key, secret_key=self.cfg.secret_key)


        # Convertir interval "5m" → TimeFrame(5, Minute)
        interval_str = self.cfg.interval
        if interval_str.endswith("m"):
            minutes = int(interval_str.replace("m",""))
            timeframe = TimeFrame(amount=minutes, unit=TimeFrameUnit.Minute)
        else:
            raise ValueError(f"Intervalo no soportado para Alpaca: {interval_str}")

        request = StockBarsRequest(
            symbol_or_symbols=self.cfg.symbol,
            timeframe=timeframe,
            start=datetime.fromisoformat(self.cfg.start_date),
            end=datetime.fromisoformat(self.cfg.end_date)
        )

        bars = client.get_stock_bars(request)
        df = bars.df.reset_index()

        # Normalización igual que en YFinance
        df = df.rename(columns={"timestamp": "datetime"})
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df.columns = [c.lower() for c in df.columns]
        df = df.sort_values("datetime").reset_index(drop=True)

        return df

    

    def cargar_datos(self):
        """
        Carga los datos desde la fuente configurada.
        Si la fuente es un CSV, lo lee desde la ruta indicada.
        Si la fuente es yfinance, los descarga.
        Guarda el resultado en self.df y lo devuelve.
        """
        if self.cfg.source == "csv":
            if not self.cfg.csv_path:
                raise ValueError("Debes indicar la ruta del CSV en cfg.csv_path")

            self.df = self._load_csv(self.cfg.csv_path)

        elif self.cfg.source == "yfinance":
            self.df = self._fetch_yfinance()

        elif self.cfg.source == "alpaca":
            self.df = self._fetch_alpaca()

        else:
            raise ValueError("Fuente de datos no válida. Usa 'csv', 'alpaca o 'yfinance'.")

        return self.df


def preprocess_data(df):

    # Limpieza y preprocesado básico de datos OHLCV.
    # Ordena por fecha
    # Convierte a datetime
    # Sustituye volumen 0 por NaN e interpola
    # Calcula retornos y log-retornos

    df  =  df.copy()

     # Convertimos la columna 'datetime' a tipo date_time y ordenamos los datos por si acasso
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.sort_values("datetime").reset_index(drop=True)

    # Limpiar volumen (solo si existe)
    if "volume" in df.columns:
        df["volume"] = df["volume"].astype(float)
        #Convierte a NaN todas las filas de la columna "volume" donde sea = 0
        df.loc[df["volume"] == 0, "volume"] = np.nan
        #Rellena los datos que NaN interpolando linealmente 
        df["volume"] = df["volume"].interpolate(method="linear")

        # Volumen normalizado 
        vol_mean = df["volume"].mean()
        vol_std = df["volume"].std()
        if vol_std == 0 or np.isnan(vol_std):
            df["volume_norm"] = 0.0
        else:
            df["volume_norm"] = (df["volume"] - vol_mean) / vol_std




    # Calcular retornos
    df["return"] = df["close"].pct_change()
    df["log_return_raw"] = np.log(df["close"])
    df["log_return_close"] = df["log_return_raw"].diff()


    # Evitamos división por cero/NaNs usando clip mínimo muy pequeño
    eps = 1e-12

    # log( close / open )
    if "open" in df.columns and "close" in df.columns:
        df["log_ret_oc"] = np.log(
            (df["close"].clip(lower=eps)) / (df["open"].clip(lower=eps))
        )

    # log( high / low )
    if "high" in df.columns and "low" in df.columns:
        df["log_ret_hl"] = np.log(
            (df["high"].clip(lower=eps)) / (df["low"].clip(lower=eps))
        )

    # log( close / high )
    if "close" in df.columns and "high" in df.columns:
        df["log_ret_ch"] = np.log(
            (df["close"].clip(lower=eps)) / (df["high"].clip(lower=eps))
        )

    # log( close / low )
    if "close" in df.columns and "low" in df.columns:
        df["log_ret_cl"] = np.log(
            (df["close"].clip(lower=eps)) / (df["low"].clip(lower=eps))
        )


    # Quitar primeras filas con NaN
    df = df.dropna().reset_index(drop=True)

    return df

# TEST:
# cfg = DataCleanerConfig(source="yfinance",symbol="QQQ",interval="15m",start_date="2025-10-01",end_date="2025-10-22")
# symbol_df = DataCleaner(cfg)
# raw_df = symbol_df.cargar_datos()

# #Preprocesar
# df_limpio = preprocess_data(raw_df)

# print(df_limpio)
# print(f"\nFilas finales: {len(df_limpio)}")


# TEST: 

if __name__ == "__main__":


    # TEST CARGAR_DATOS
    cfg = DataCleanerConfig(source="yfinance",symbol="BTC-USD",interval="1h",start_date="2025-10-20", end_date= "2025-10-21")
    #   TEST CARGAR_DATOS
    symbol_data = DataCleaner(cfg)
    datos = symbol_data.cargar_datos()

    print(datos)
    print(f"\nFilas descargadas: {len(datos)}")
   

# TEST PREPROCESS_DATA:
   

