from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# criar escopo
scope_name = "openrouter"
w.secrets.create_scope(scope=scope_name)

# criar secret dentro do escopo
w.secrets.put_secret(scope_name, "minha-key", string_value="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")