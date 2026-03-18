# core/comparacao_2amostras.py
import math
import numpy as np
from scipy.stats import f, t
from typing import Dict, Any, Tuple

def teste_f_variancias(dados1: np.ndarray, dados2: np.ndarray, alpha: float = 0.05) -> Dict[str, Any]:
    """
    Teste F para igualdade de variâncias (bilateral adaptado).
    Retorna estatística F, p-valor (aproximado) e decisão.
    """
    var1 = np.var(dados1, ddof=1)
    var2 = np.var(dados2, ddof=1)
    n1, n2 = len(dados1), len(dados2)
    
    # Coloca a maior variância no numerador
    if var1 >= var2:
        F_calc = var1 / var2
        gl_num = n1 - 1
        gl_den = n2 - 1
        var_maior = "grupo1"
    else:
        F_calc = var2 / var1
        gl_num = n2 - 1
        gl_den = n1 - 1
        var_maior = "grupo2"
    
    # p-valor bilateral (multiplica por 2 a cauda superior)
    p_valor = 2 * (1 - f.cdf(F_calc, gl_num, gl_den))
    # Limita a 1
    p_valor = min(p_valor, 1.0)
    
    # Decisão (H0: variâncias iguais)
    rejeita_h0 = p_valor < alpha
    
    return {
        "F_calc": F_calc,
        "gl_num": gl_num,
        "gl_den": gl_den,
        "p_valor": p_valor,
        "rejeita_h0": rejeita_h0,
        "variancias_iguais": not rejeita_h0,  # para uso posterior
        "var_maior": var_maior
    }

def teste_t_medias(dados1: np.ndarray, dados2: np.ndarray, variancias_iguais: bool = True, alpha: float = 0.05) -> Dict[str, Any]:
    """
    Teste t para comparação de médias de duas amostras independentes.
    
    Args:
        dados1, dados2: arrays
        variancias_iguais: se True, usa teste pooled; se False, usa Welch
        alpha: nível de significância
    
    Returns:
        dicionário com estatísticas e conclusão
    """
    n1, n2 = len(dados1), len(dados2)
    media1, media2 = np.mean(dados1), np.mean(dados2)
    var1, var2 = np.var(dados1, ddof=1), np.var(dados2, ddof=1)
    desv1, desv2 = np.std(dados1, ddof=1), np.std(dados2, ddof=1)
    
    diferenca = media1 - media2
    
    if variancias_iguais:
        # Pooled
        sp2 = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        erro_padrao = math.sqrt(sp2) * math.sqrt(1/n1 + 1/n2)
        gl = n1 + n2 - 2
        metodo = "t pooled (variâncias iguais)"
    else:
        # Welch
        erro_padrao = math.sqrt(var1/n1 + var2/n2)
        # Graus de liberdade aproximados (Welch-Satterthwaite)
        A = var1 / n1
        B = var2 / n2
        gl = (A + B)**2 / (A**2/(n1-1) + B**2/(n2-1))
        metodo = "t Welch (variâncias diferentes)"
    
    t_calc = diferenca / erro_padrao
    t_critico = t.ppf(1 - alpha/2, gl)
    p_valor = 2 * (1 - t.cdf(abs(t_calc), gl))
    
    rejeita_h0 = abs(t_calc) > t_critico
    
    return {
        "metodo": metodo,
        "t_calc": t_calc,
        "t_critico": t_critico,
        "p_valor": p_valor,
        "gl": gl,
        "rejeita_h0": rejeita_h0,
        "media1": media1,
        "media2": media2,
        "diferenca": diferenca,
        "n1": n1,
        "n2": n2
    }

def comparacao_completa(dados1: np.ndarray, dados2: np.ndarray, alpha: float = 0.05) -> Dict[str, Any]:
    """
    Executa todo o protocolo: teste F para variâncias, depois teste t adequado.
    Retorna resultados consolidados.
    """
    res_f = teste_f_variancias(dados1, dados2, alpha)
    var_iguais = res_f["variancias_iguais"]
    res_t = teste_t_medias(dados1, dados2, var_iguais, alpha)
    
    return {
        "teste_f": res_f,
        "teste_t": res_t,
        "alpha": alpha,
        "conclusao_f": "Variâncias iguais" if var_iguais else "Variâncias diferentes",
        "conclusao_medias": "Médias diferentes" if res_t["rejeita_h0"] else "Médias iguais"
    }