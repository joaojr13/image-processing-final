"""Valores usados no treino. Alterações neste arquivo valem para novos treinos."""

EPOCAS = 200                 # Quantas vezes a rede percorre os exemplos de treino.
SEMENTE = 42                # Controla a aleatoriedade para facilitar comparações.
TAXA_APRENDIZADO = 0.03      # Tamanho dos ajustes realizados pelo otimizador.
DECAIMENTO_PESOS = 0.001     # Regularização que reduz a magnitude dos pesos.
EPSILON = 1e-8              # Constante de estabilidade numérica do AdamW.
LOTE_EXTRACAO = 8           # Rostos enviados juntos à FaceNet congelada.
