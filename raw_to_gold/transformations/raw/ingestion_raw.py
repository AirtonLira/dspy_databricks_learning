
from pyspark import pipelines as dp
from pyspark.sql.functions import *


# ========================================
# RAW LAYER - Ingestão de Dados Brutos
# ========================================
@dp.table(
    name="raw_b_2_w_reviews",  
    comment="Tabela raw da ingestão do csv de avaliações dos clientes"
)
def ingestion_raw_layer():
     return (
        spark.read
        .format("csv")
        .option("header", "true")         
        .option("inferSchema", "true")
        .load("/Volumes/workspace/b2w/dspy_databricks_learning/b2w_reviews.csv")
    )
