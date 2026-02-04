# ===============================================================================
# 1. SETUP E VARIÁVEIS GLOBAIS
# ===============================================================================
import dspy
import pandas as pd
import json
import time
import os
import sys
from pyspark.sql.types import StringType, StructType, StructField
from pyspark.sql.functions import col, pandas_udf, from_json
from dspy import InputField, OutputField, Signature

# --- PATH CONFIG ---
REPO_ROOT_PATH = "/Workspace/Users/airtonlirajr@gmail.com/dspy_databricks_learning/raw_to_gold"

# --- AUTH CONFIG (Captura no Driver) ---
try:
    CTX_HOST = spark.conf.get("spark.databricks.workspaceUrl")
    if not CTX_HOST.startswith("http"):
        CTX_HOST = f"https://{CTX_HOST}"
    CTX_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
except Exception:
    CTX_HOST = None
    CTX_TOKEN = None

# ===============================================================================
# 3. UDF com Blindagem Total (Try/Catch no Loop)
# ===============================================================================
@pandas_udf(StringType())
def extract_reason_udf(texts: pd.Series) -> pd.Series:
    # --- Configurações do Worker ---
    import sys, os, time
    
    # 1. Path
    if REPO_ROOT_PATH not in sys.path:
        sys.path.append(REPO_ROOT_PATH)
    
    # 2. Auth Injection
    if CTX_HOST: os.environ["DATABRICKS_HOST"] = CTX_HOST
    if CTX_TOKEN: os.environ["DATABRICKS_TOKEN"] = CTX_TOKEN
    
    from utils.dspy_config import DspyConfig
    
    results = []
    
    # Instancia Configuração
    try:
        dspy_silver = DspyConfig()
    except Exception as e:
        # Se falhar na config, retorna erro para todas as linhas do batch
        err_msg = json.dumps({"reason": "ERRO_CONFIG", "error_detail": str(e)})
        return pd.Series([err_msg] * len(texts))
    
    # --- LOOP DE PROCESSAMENTO BLINDADO ---
    for text in texts.tolist():
        row_result = {
            "reason": None,
            "status": "processed",
            "error_detail": None
        }

        # BLINDAGEM 1: Proteção Global contra qualquer falha na linha
        try:
            # Validação de Entrada
            text_safe = str(text) if text is not None else ""
            text_clean = text_safe.strip()

            # Se for muito curto (< 4 chars), ignora para evitar erro "0 tokens"
            if len(text_clean) < 4:
                row_result["status"] = "skipped_short"
                results.append(json.dumps(row_result))
                continue

            # Chamada DSPy
            extractor = dspy_silver.predict(text_safe)
                
            row_result["reason"] = extractor
            row_result["status"] = "success"

        except Exception as e:
            # BLINDAGEM 2: Captura o erro do LiteLLM/OpenRouter aqui
            # Isso impede que a Pipeline pare. O erro vai para a coluna 'error_detail'
            row_result["reason"] = "ERRO_LLM"
            row_result["status"] = "error"
            row_result["error_detail"] = str(e)[0:1000] # Trunca mensagem longa
        
        # Adiciona o resultado (seja sucesso ou erro tratado) à lista
        results.append(json.dumps(row_result))
        
        # Rate Limit
        time.sleep(0.05) 
            
    return pd.Series(results)

# ===============================================================================
# 4. Pipeline Spark Declarative Pipelines
# ===============================================================================
from pyspark import pipelines as dp

@dp.materialized_view(
    name="silver_b_2_w_reviews",
    comment="Tabela final com motivos extraídos via LLM."
)
def silver_pipeline_logic():
    # Lê da Bronze (Verifique o nome da sua tabela de origem!)
    df_in = spark.read.table("bronze_b_2_w_reviews").limit(50)
    
    # Aplica UDF
    df_processed = df_in.withColumn(
        "sentiment", 
        extract_reason_udf(col("review_text"))
    )
    
    return df_processed
