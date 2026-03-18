import numpy as np

def limpar_dados(dados_brutos):
    try:
        return [float(x) for x in dados_brutos if x]
    except:
        return None

def validar_tamanho(dados):
    if dados is None:
        return False, "Erro na conversão dos dados."
    if len(dados) < 2:
        return False, "É necessário pelo menos dois valores."
    return True, ""

def detectar_outliers_iqr(dados):
    dados = np.array(dados)
    q1 = np.percentile(dados, 25)
    q3 = np.percentile(dados, 75)
    iqr = q3 - q1

    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr

    outliers = [(x < limite_inferior or x > limite_superior) for x in dados]

    return (limite_inferior, limite_superior), outliers