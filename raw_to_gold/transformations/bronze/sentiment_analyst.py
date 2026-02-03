from pyspark import pipelines as dp
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType, IntegerType

def _classify_sentiment(rating: int) -> str:
    if rating > 3:
        return "positivo"
    elif rating < 3:
        return "negativo"
    return "neutro"

classify_sentiment_udf = udf(_classify_sentiment, StringType())

@dp.materialized_view(
    name="_bronze_b_2_w_reviews",
    comment="Base de reviews pronta com avaliação de sentimento pela LLM"
)
def bronze_b2w_reviews():
    df = spark.read.table("LIVE._b_2_w_reviews")
    df_clean = df.filter(col("overall_rating_int") > 1)
    df_clean = df_clean.filter(col("review_text").isNotNull())
    df_clean = df_clean.withColumn("sentiment", classify_sentiment_udf(col("overall_rating_int")))
    return df_clean
