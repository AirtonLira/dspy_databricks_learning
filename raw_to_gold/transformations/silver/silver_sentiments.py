import sys
import os

# --- CONFIGURAÇÃO MANUAL DO CAMINHO (OBRIGATÓRIO NO COMMUNITY) ---
project_root = "/Workspace/Users/airtonlirajr@gmail.com/dspy_databricks_learning/raw_to_gold"

if project_root not in sys.path:
    sys.path.append(project_root)

from pyspark.sql.functions import pandas_udf, col, current_timestamp, lit
from pyspark.sql.types import StringType
from utils.dspy_config import DspyConfig
import pandas as pd
import time
import mlflow
import dspy

# ============================
# 2. Definir Signature (DSPy)
# ============================
class ExtractSentimentReason(dspy.Signature):
    """
    Analista de reviews brasileiro que identifica o motivo principal do sentimento.
    Retorna frase curta em português sobre produto ou ocasião que gerou o sentimento.
    """
    review_text: str = dspy.InputField(desc="Texto completo da avaliação do cliente")
    sentiment: str = dspy.InputField(desc="Sentimento já classificado: positivo, negativo ou neutro")
    
    reason: str = dspy.OutputField(
        desc="Motivo principal do sentimento em UMA frase curta (máximo 10 palavras). "
             "Exemplos: 'produto chegou quebrado', 'entrega foi rápida'. Se vago, responda 'motivo não claro'."
    )

# Opção A: Predict simples
dspy_config = DspyConfig()
dspy_config.test_api()

extract_reason = dspy_config.dspy.Predict(ExtractSentimentReason)
# ============================
# 3. Função auxiliar para processar
# ============================
def process_single_review(review_text: str, sentiment: str) -> str:
    try:
        response = extract_reason(review_text=review_text, sentiment=sentiment)
        return response.reason
    except Exception as e:
        return f"erro_dspy: {str(e)[:30]}"

# ============================
# 4. Pandas UDF com Métricas
# ============================
@pandas_udf(StringType())
def extract_sentiment_reason_dspy(texts: pd.Series, sentiments: pd.Series) -> pd.Series:
    results = []
    
    for text, sentiment in zip(texts, sentiments):
        try:
            reason = process_single_review(text, sentiment)
            
            # Incrementa métricas de sucesso/erro
            if "erro_dspy" in reason:
                error_count.add(1)
            else:
                success_count.add(1)
                
            results.append(reason)
            time.sleep(0.2) # Rate limit
            
        except Exception as e:
            error_count.add(1)
            results.append(f"erro_batch: {str(e)[:200]}")
    
    return pd.Series(results)

# ============================
# 5. Pipeline Gold com Auditoria
# ============================

import dlt
import mlflow
from pyspark.sql.functions import col, current_timestamp, lit

@dlt.table(
    name="_gold_b_2_reviews",
    comment="Reviews com análise de motivo do sentimento via DSPy + OpenRouter"
)
def gold_b_2_reviews():
    # Definir experimento no MLflow
    try:
        mlflow.set_experiment("/Shared/Sentiment_Analysis_Metrics")
    except Exception as e:
        print(f"Não foi possível setar o experimento MLflow: {e}")

    # 1. Leitura da Bronze
    df_in = (
        spark.read.table("LIVE._bronze_b_2_w_reviews")
        .select("reviewer_id", "review_text", "sentiment")
        .limit(50)  # Remova ou altere conforme necessidade
    )

    # 2. Aplicar extração de motivo de sentimento (função definida separadamente)
    df_out = (
        df_in
        .withColumn(
            "sentiment_reason",
            extract_sentiment_reason_dspy(col("review_text"), col("sentiment"))
        )
        .withColumn("processing_timestamp", current_timestamp())
        .withColumn("model_version", lit("liquid-lfm-2.5-1.2b-free"))
    )

    # 3. Registrar métricas no MLflow
    # Observação: métricas só serão computadas após ação de escrita
    try:
        with mlflow.start_run(run_name="Pipeline_Gold_DSPy"):
            mlflow.log_param("input_rows", df_out.count())
            mlflow.log_param("model_version", "liquid-lfm-2.5-1.2b-free")
    except Exception as e:
        print(f"Erro ao logar métricas no MLflow: {e}")

    # 4. Retornar DataFrame materializado
    return df_out