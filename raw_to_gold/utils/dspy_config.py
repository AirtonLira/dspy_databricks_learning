import dspy
from dspy import InputField, OutputField, Signature
from pyspark.sql.functions import col, pandas_udf, current_timestamp, lit
from pyspark.sql.types import StringType
from databricks.sdk.runtime import dbutils
import pandas as pd
import time
import mlflow


class DspyConfig:
    """
    Classe para configuração de métricas, assinatura DSPy e análise de sentimento de reviews.
    Permite reutilização dos objetos e métodos em outros códigos.
    """

    def __init__(self):
        self.success_count = 0
        self.error_count = 0

        # Pegar chave do secret
        self.OPENROUTER_API_KEY = dbutils.secrets.get("openrouter", "api_key") 
        self.dspy = dspy

        # Configurar DSPy localmente
        self.lm_config = dspy.LM(
            model="openrouter/liquid/lfm-2.5-1.2b-instruct:free",
            api_key=self.OPENROUTER_API_KEY,
            max_tokens=256
        )
        self.dspy.configure(lm=self.lm_config)

        # Definir assinatura para análise de sentimento
        self.sentiment_signature = Signature(
            inputs=[InputField(name="texto", dtype=str, description="Texto para análise de sentimento")],
            outputs=[OutputField(name="sentimento", dtype=str, description="Sentimento: positivo, negativo ou neutro")]
        )

    def predict(self, texto, signature=None):
        try:
            # Usar assinatura de sentimento se não for fornecida outra
            if signature is None:
                signature = self.sentiment_signature

            # Criar preditor DSPy com a assinatura
            predictor = self.dspy.Predict(signature)

            # Realizar predição
            result = predictor(texto=texto)

            # Extrair sentimento do resultado
            sentimento = result.sentimento if hasattr(result, "sentimento") else None

            self.success_count += 1
            return sentimento
        except Exception as e:
            self.error_count += 1
            return None
            if signature is None:
                signature = self.sentiment_signature
            predictor