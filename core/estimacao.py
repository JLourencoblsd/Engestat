# core/estimacao.py
import math
import numpy as np
from scipy.stats import norm, t, chi2
from typing import Dict, Any, Optional

def ic_media(dados: np.ndarray, alpha: float = 0.05, sigma_conhecido: Optional[float] = None) -> Dict[str, Any]:
    """
    Intervalo de confiança para a média populacional.
    
    Args:
        dados: array de dados
        alpha: nível de significância (1 - confiança)
        sigma_conhecido: desvio padrão populacional conhecido (opcional)
    
    Returns:
        dicionário com limites, método, e estatísticas usadas
    """
    n = len(dados)
    media = np.mean(dados)
    S = np.std(dados, ddof=1)
    
    if sigma_conhecido is not None:
        z = norm.ppf(1 - alpha/2)
        erro = z * sigma_conhecido / math.sqrt(n)
        metodo = "Normal (σ conhecido)"
        sigma_usado = sigma_conhecido
    else:
        t_val = t.ppf(1 - alpha/2, n - 1)
        erro = t_val * S / math.sqrt(n)
        metodo = "t-Student (σ desconhecido)"
        sigma_usado = S
    
    li = media - erro
    ls = media + erro
    
    return {
        "parametro": "média (μ)",
        "metodo": metodo,
        "n": n,
        "media": media,
        "desvio_usado": sigma_usado,
        "alpha": alpha,
        "li": li,
        "ls": ls,
        "erro": erro
    }

def ic_desvio(dados: np.ndarray, alpha: float = 0.05) -> Dict[str, Any]:
    """
    Intervalo de confiança para o desvio padrão populacional (método qui-quadrado).
    Inclui aproximação normal para n > 30.
    """
    n = len(dados)
    S = np.std(dados, ddof=1)
    gl = n - 1
    
    chi2_inf = chi2.ppf(1 - alpha/2, gl)
    chi2_sup = chi2.ppf(alpha/2, gl)
    
    li = math.sqrt((gl * S**2) / chi2_inf)
    ls = math.sqrt((gl * S**2) / chi2_sup)
    
    resultado = {
        "parametro": "desvio padrão (σ)",
        "metodo": "Qui-Quadrado",
        "n": n,
        "S": S,
        "alpha": alpha,
        "li": li,
        "ls": ls
    }
    
    # Aproximação normal para amostras grandes
    if n > 30:
        z = norm.ppf(1 - alpha/2)
        li_n = S / (1 + z / math.sqrt(2 * gl))
        ls_n = S / (1 - z / math.sqrt(2 * gl))
        resultado["aprox_normal"] = {"li": li_n, "ls": ls_n}
    
    return resultado

def ic_variancia(dados: np.ndarray, alpha: float = 0.05) -> Dict[str, Any]:
    """
    Intervalo de confiança para a variância populacional.
    """
    n = len(dados)
    S2 = np.var(dados, ddof=1)
    gl = n - 1
    
    chi2_inf = chi2.ppf(1 - alpha/2, gl)
    chi2_sup = chi2.ppf(alpha/2, gl)
    
    li = (gl * S2) / chi2_inf
    ls = (gl * S2) / chi2_sup
    
    return {
        "parametro": "variância (σ²)",
        "metodo": "Qui-Quadrado",
        "n": n,
        "S2": S2,
        "alpha": alpha,
        "li": li,
        "ls": ls
    }