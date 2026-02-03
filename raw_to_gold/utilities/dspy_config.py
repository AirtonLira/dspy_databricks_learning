class DspyConfig:
    """
    Classe para configuração de métricas, assinatura DSPy e análise de sentimento de reviews.
    Permite reutilização dos objetos e métodos em outros códigos.
    """

    def __init__(self):
        # 1. Configuração de Métricas (Accumulators)
        self.success_count = spark.sparkContext.accumulator(0)
        self.error_count = spark.sparkContext.accumulator(0)

        # 2. Definir Signature (DSPy)
        class ExtractSentimentReason(dspy.Signature):
            """
            Analista de reviews brasileiro que identifica o motivo principal do sentimento.
            Retorna frase curta em português sobre produto ou ocasião que gerou o sentimento.
            """
            review_text: str = dspy.InputField(desc="Texto completo da avaliação do cliente")
            sentiment: str = dspy.OutputField(desc="Sentimento para classificar apartir do texto: positivo, negativo ou neutro")
        self.ExtractSentimentReason = ExtractSentimentReason

        # Pegar chave do secret
        self.OPENROUTER_API_KEY = dbutils.secrets.get("openrouter", "api_key") 

        # Configurar DSPy localmente
        self.lm_config = dspy.LM(
            model="openrouter/liquid/lfm-2.5-1.2b-instruct:free",
            api_key=self.OPENROUTER_API_KEY,
            max_tokens=256
        )
        dspy.configure(lm=self.lm_test)

    def test_api(self):
        print("Iniciando teste de chamada API...")
        try:
            # Definir a classe inline só para teste
            class TestSignature(dspy.Signature):
                texto = dspy.InputField()
                resposta = dspy.OutputField()
            
            predictor = dspy.Predict(TestSignature)
            resultado = predictor(texto="Diga 'Olá Mundo' em português.")
            
            print("\n✅ SUCESSO! A API respondeu:")
            print(resultado.resposta)
            self.success_count += 1
        except Exception as e:
            print(f"\n❌ ERRO NA API: {str(e)}")
            self.error_count += 1