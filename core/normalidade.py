# core/normalidade.py
import numpy as np
from scipy import stats
from typing import Dict, Any, Tuple

def shapiro_wilk(x: np.ndarray) -> Dict[str, Any]:
    """
    Teste de Shapiro-Wilk para normalidade.
    
    Args:
        x: array de dados
        
    Returns:
        dicionário com estatística, p-valor, e conclusão lógica
    """
    stat, p = stats.shapiro(x)
    return {
        'teste': 'Shapiro-Wilk',
        'estatistica': stat,
        'pvalor': p,
        'normal': p > 0.05,
        'interpretacao': 'dados normais' if p > 0.05 else 'dados não normais'
    }

def kolmogorov_smirnov_lilliefors(x: np.ndarray) -> Dict[str, Any]:
    """
    Teste de Kolmogorov-Smirnov com parâmetros estimados (Lilliefors).
    
    Args:
        x: array de dados
        
    Returns:
        dicionário com estatística, p-valor, e conclusão
    """
    from scipy.stats import kstest
    media = np.mean(x)
    desvio = np.std(x, ddof=1)
    stat, p = kstest(x, 'norm', args=(media, desvio))
    return {
        'teste': 'Kolmogorov-Smirnov (Lilliefors)',
        'estatistica': stat,
        'pvalor': p,
        'normal': p > 0.05,
        'interpretacao': 'dados normais' if p > 0.05 else 'dados não normais'
    }

def chi_square_goodness_of_fit(x: np.ndarray, n_classes: int = 8) -> Dict[str, Any]:
    """
    Teste Qui-Quadrado de aderência à normal.
    
    Args:
        x: array de dados
        n_classes: número de classes para o histograma
        
    Returns:
        dicionário com estatística, p-valor, graus de liberdade e conclusão
    """
    # Ordena e define limites
    x_sorted = np.sort(x)
    n = len(x)
    minimo = np.min(x)
    maximo = np.max(x)
    limites = np.linspace(minimo, maximo, n_classes + 1)
    
    # Frequências observadas
    freq_obs, _ = np.histogram(x, bins=limites)
    
    # Parâmetros estimados
    media = np.mean(x)
    desvio = np.std(x, ddof=1)
    
    # Probabilidades teóricas para cada classe
    prob_teorica = []
    for i in range(n_classes):
        a = stats.norm.cdf(limites[i], media, desvio)
        b = stats.norm.cdf(limites[i+1], media, desvio)
        prob_teorica.append(b - a)
    
    freq_esp = n * np.array(prob_teorica)
    
    # Evitar divisão por zero (se alguma freq_esp for muito pequena, combinamos classes?)
    # Por simplicidade, assumimos que o usuário escolhe um número adequado de classes.
    estatistica = np.sum((freq_obs - freq_esp) ** 2 / freq_esp)
    
    # Graus de liberdade: k - 1 - 2 (dois parâmetros estimados)
    df = n_classes - 1 - 2
    p = 1 - stats.chi2.cdf(estatistica, df)
    
    return {
        'teste': 'Qui-Quadrado de Aderência',
        'estatistica': estatistica,
        'pvalor': p,
        'df': df,
        'normal': p > 0.05,
        'interpretacao': 'dados normais' if p > 0.05 else 'dados não normais'
    }

def qq_plot_data(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retorna quantis teóricos e dados ordenados para um Q-Q plot.
    
    Returns:
        (quantis_teoricos, dados_ordenados)
    """
    from scipy import stats
    x_sorted = np.sort(x)
    n = len(x)
    # Probabilidades (posições de plotagem) usando (i-0.5)/n
    p = (np.arange(1, n+1) - 0.5) / n
    quantis_teoricos = stats.norm.ppf(p)
    return quantis_teoricos, x_sorted