from databricks.sdk import WorkspaceClient
import os

w = WorkspaceClient()

# criar escopo
scope_name = "openrouter"
w.secrets.create_scope(scope=scope_name)

# criar secret dentro do escopo
w.secrets.put_secret(scope_name, "minha-key", string_value=os.getenv("OPENROUTER_API_KEY)