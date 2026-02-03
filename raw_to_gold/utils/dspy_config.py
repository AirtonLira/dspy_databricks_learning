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

        # 2. Definir Signature (DSPy)
        class ExtractSentimentReason(dspy.Signature):
            """
            Analista de reviews brasileiro que identifica o motivo principal do sentimento.
            Retorna frase curta em português sobre produto ou ocasião que gerou o sentimento.
            """
            review_text: str = dspy.InputField(desc="Texto completo da avaliação do cliente")
            sentiment: str = dspy.OutputField(desc="Sentimento para classificar apartir do texto: positivo, negativo ou neutro")
        self.ExtractSentimentReason = ExtractSentimentReason

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

    def test_api(self):
        print("Iniciando teste de chamada API...")
        try:
            # Definir a classe inline só para teste
            class TestSignature(dspy.Signature):
                texto = self.dspy.InputField()
                resposta = self.dspy.OutputField()
            
            predictor = self.dspy.Predict(TestSignature)
            resultado = predictor(texto="Diga 'Olá Mundo' em português.")
            
            print("\n✅ SUCESSO! A API respondeu:")
            print(resultado.resposta)
            self.success_count += 1
        except Exception as e:
            print(f"\n❌ ERRO NA API: {str(e)}")
            self.error_count += 1