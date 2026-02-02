# Databricks notebook source
# MAGIC %md
# MAGIC Criação de volumes:

# COMMAND ----------

spark.sql("CREATE VOLUME IF NOT EXISTS sandbox.vendas.dspy_databricks_learning;")

# COMMAND ----------

# MAGIC %md
# MAGIC Dropar tabela:

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE sandbox.vendas.b_2_w_reviews;

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE VOLUME IF NOT EXISTS workspace.b2w.dspy_databricks_learning;

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE workspace.b2w.b_2_w_reviews_with_sentiment;

# COMMAND ----------

df = spark.sql("SELECT * FROM workspace.b2w.b_2_w_reviews_with_sentiment")
df.summary().show()  
