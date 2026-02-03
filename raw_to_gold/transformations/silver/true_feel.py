import dspy
from dspy import InputField, OutputField, Signature
from pyspark.sql.functions import col, pandas_udf, current_timestamp, lit
from pyspark.sql.types import StringType
from pyspark import pipelines as dp 
import pandas as pd
import time
import mlflow

# ==========================================
# 1. Configuração de Métricas (Accumulators)
# ==========================================
# Estes contadores funcionam dentro do cluster Spark
success_count = spark.sparkContext.accumulator(0)
error_count = spark.sparkContext.accumulator(0)

# ============================
# 2. Definir Signature (DSPy)
# ============================
class ExtractSentimentReason(Signature):
    """
    Analista de reviews brasileiro que identifica o motivo principal do sentimento.
    Retorna frase curta em português sobre produto ou ocasião que gerou o sentimento.
    """
    review_text: str = InputField(desc="Texto completo da avaliação do cliente")
    sentiment: str = InputField(desc="Sentimento já classificado: positivo, negativo ou neutro")
    
    reason: str = OutputField(
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

@dp.materialized_view(
    name="_gold_b_2_reviews",
    comment="Reviews com análise de motivo do sentimento via DSPy + OpenRouter"
)
def gold_reviews_with_sentiment_reason():
    # Iniciar monitoramento no MLflow
    mlflow.set_experiment("/Shared/Sentiment_Analysis_Metrics")
    
    with mlflow.start_run(run_name="Pipeline_Gold_DSPy"):
        # 1. Leitura da Bronze
        df_in = (
            spark.read.table("LIVE._bronze_b_2_w_reviews")
            .select("reviewer_id", "review_text", "sentiment")
            .limit(50) # Remova ou altere o limit conforme necessidade
        )

        # 2. Processamento com Adição de Metadados de Auditoria
        df_out = df_in.withColumn(
            "sentiment_reason",
            extract_sentiment_reason_dspy(col("review_text"), col("sentiment"))
        ).withColumn(
            "processing_timestamp", current_timestamp() # Quando foi processado
        ).withColumn(
            "model_version", lit("liquid-lfm-2.5-1.2b-free") # Versão do modelo
        )

        # 3. Log de métricas finais no MLflow (visível após a execução)
        # Nota: Como o Spark é lazy, o valor real dos accumulators 
        # só estará correto após a ação de escrita ser disparada pelo Dataprix.
        
        return df_out
