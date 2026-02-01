from pyspark import pipelines as dp
from pyspark.sql.functions import *

@dp.table(                      
    name="_b_2_w_reviews",
    comment="Tabela raw da ingestão do csv de avaliações dos clientes, com coluna de sentimento"
)
def positive_or_negative_raw():
    df = spark.read.table("sandbox.vendas.b_2_w_reviews")
    df = df.withColumn(
        "sentiment",
        when(col("overall_rating") >= 4, lit("positivo"))
        .when(col("overall_rating") <= 2, lit("negativo"))
        .otherwise(lit("neutro"))
    )
    return df