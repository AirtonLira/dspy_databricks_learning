import dspy
import pandas as pd
import json
import time
from pyspark.sql.types import StringType
from utils.dspy_config import DspyConfig
from pyspark.sql.functions import col, pandas_udf, from_json, schema_of_json, lit, avg
from pyspark.sql.types import StringType, StructType, StructField, FloatType, IntegerType
from dspy import InputField, OutputField, Signature
from pyspark import pipelines as dp

# ==============================================================================
# 1. Configuração de Segredos (Broadcast para Segurança)
# ==============================================================================
dspy_config = DspyConfig()

# ==============================================================================
# 2. Definição do Modelo DSPy (Signature)
# ==============================================================================
class ExtractSentimentReason(Signature):
    """
    Analista que identifica o motivo do sentimento em reviews de e-commerce.
    """
    review_text: str = InputField(desc="Texto completo da avaliação")
    sentiment: str = InputField(desc="Sentimento classificado (Pos/Neg/Neutro)")
    
    reason: str = OutputField(
        desc="Motivo curto (max 10 palavras). Ex: 'entrega atrasada', 'produto defeituoso'."
    )

# ==============================================================================
# 3. Lógica de Processamento (Worker Side)
# ==============================================================================
def process_single_review(review_text: str, sentiment: str, lm_instance):
    """Processa um único texto e calcula métricas de engenharia."""
    start_time = time.time()
    result = {
        "reason": None,
        "latency_sec": 0.0,
        "word_count": 0,
        "status": "error",
        "error_msg": None
    }
    
    try:
        # Context manager garante que usamos a configuração correta
        with dspy.context(lm=lm_instance):
            extractor = dspy.Predict(ExtractSentimentReason)
            response = extractor(review_text=review_text, sentiment=sentiment)
            
            # Sucesso
            reason_text = response.reason
            result["reason"] = reason_text
            result["word_count"] = len(reason_text.split())
            result["status"] = "success"
            
    except Exception as e:
        # Captura erro completo (aumentado limite para 500 chars)
        result["error_msg"] = str(e)[:500]
        result["reason"] = "ERRO_PROCESSAMENTO"

    # Calcula latência final
    result["latency_sec"] = round(time.time() - start_time, 4)
    
    return json.dumps(result) # Retorna JSON string para o Spark parsear depois

# ==============================================================================
# 4. Pandas UDF Otimizada
# ==============================================================================
@pandas_udf(StringType())
def extract_reason_udf(texts: pd.Series, sentiments: pd.Series) -> pd.Series:
    results = []

    dspy_config = DspyConfig()
    dspy_config.test_api()

    # 2. Configura o modelo uma vez por batch
    lm = dspy.LM(
        model="openrouter/liquid/lfm-2.5-1.2b-instruct:free",
        api_key=dspy_config.OPENROUTER_API_KEY,
        max_tokens=128
    )
    
    # 3. Converte para listas Python (CORREÇÃO DO BUG DA LETRA "A")
    # Isso garante iteração item a item, não caractere a caractere
    text_list = texts.tolist()
    sent_list = sentiments.tolist()
    
    for text, sentiment in zip(text_list, sent_list):
        if not text:
            results.append(json.dumps({"status": "skipped", "reason": None}))
            continue
            
        json_res = process_single_review(text, sentiment, lm)
        results.append(json_res)
        
        # Rate Limit simples 
        time.sleep(0.2) 
            
    return pd.Series(results)

# ==============================================================================
# 5. Pipeline Gold Layer
# ==============================================================================
@dp.materialized_view(
    name="gold_reviews_reason_final",
    comment="Tabela final com motivos extraídos via LLM e métricas de performance."
)
def gold_pipeline_logic():
    # 1. Leitura Conectada (Isso gera a linha no gráfico!)
    df_in = spark.read.table("silver_b_2_reviews").limit(50) 
    
    # 2. Aplicação da IA
    df_processed = df_in.withColumn(
        "dspy_json_raw", 
        extract_reason_udf(col("review_text"), col("sentiment"))
    )
    
    # 3. Definição do Schema do JSON para parsing
    json_schema = StructType([
        StructField("reason", StringType(), True),
        StructField("latency_sec", FloatType(), True),
        StructField("word_count", IntegerType(), True),
        StructField("status", StringType(), True),
        StructField("error_msg", StringType(), True)
    ])
    
    # 4. Parseamento e Limpeza final
    return (
        df_processed
    )
