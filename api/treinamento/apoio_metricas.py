"""Compara nomes reais e previstos. Não executa nem treina redes."""
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def calcular_metricas(rotulos_reais, rotulos_previstos, nomes):
    rotulos = list(range(len(nomes)))
    precisao, recall, f1, suporte = precision_recall_fscore_support(
        rotulos_reais, rotulos_previstos, labels=rotulos, zero_division=0,
    )
    matriz = confusion_matrix(rotulos_reais, rotulos_previstos, labels=rotulos)

    por_pessoa = []
    for indice, nome in enumerate(nomes):
        por_pessoa.append({
            'nome': nome,
            'precisao': float(precisao[indice]),
            'recall': float(recall[indice]),
            'f1': float(f1[indice]),
            'quantidade': int(suporte[indice]),
        })

    return {
        'acuracia': float(accuracy_score(rotulos_reais, rotulos_previstos)),
        'precisao_macro': float(precisao.mean()),
        'recall_macro': float(recall.mean()),
        'f1_macro': float(f1.mean()),
        'matriz_confusao': matriz.tolist(),
        'por_pessoa': por_pessoa,
    }
