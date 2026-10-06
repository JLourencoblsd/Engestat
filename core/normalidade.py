# core/normalidade.py
import numpy as np
from scipy import stats
from typing import Dict, Any, Tuple

def shapiro_wilk(x: np.ndarray) -> Dict[str, Any]:
    """
    Teste de Shapiro-Wilk para normalidade.
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
    Teste de Lilliefors (Kolmogorov-Smirnov com parâmetros estimados da amostra).
    Requer a biblioteca statsmodels instalada.
    """
    from statsmodels.stats.diagnostic import lilliefors
    
    stat, p = lilliefors(x, dist='norm')
    return {
        'teste': 'Kolmogorov-Smirnov (Lilliefors)',
        'estatistica': stat,
        'pvalor': p,
        'normal': p > 0.05,
        'interpretacao': 'dados normais' if p > 0.05 else 'dados não normais'
    }

def chi_square_goodness_of_fit(x: np.ndarray, n_classes: int = 8) -> Dict[str, Any]:
    """
    Teste Qui-Quadrado de aderência à distribuição Normal.
    """
    n = len(x)
    minimo = np.min(x)
    maximo = np.max(x)
    
    # Limites das classes para contagem dos observados
    limites = np.linspace(minimo, maximo, n_classes + 1)
    freq_obs, _ = np.histogram(x, bins=limites)
    
    media = np.mean(x)
    desvio = np.std(x, ddof=1)
    
    # Limites estendidos (-inf e +inf) para cobrir 100% da área da curva teórica
    limites_teoricos = limites.copy()
    limites_teoricos[0] = -np.inf
    limites_teoricos[-1] = np.inf
    
    prob_teorica = []
    for i in range(n_classes):
        a = stats.norm.cdf(limites_teoricos[i], media, desvio)
        b = stats.norm.cdf(limites_teoricos[i+1], media, desvio)
        prob_teorica.append(b - a)
    
    freq_esp = n * np.array(prob_teorica)
    
    # Estatística do Qui-Quadrado
    estatistica = np.sum((freq_obs - freq_esp) ** 2 / freq_esp)
    
    # Graus de liberdade: k - 1 - 2 (2 parâmetros estimados: média e desvio)
    df = n_classes - 1 - 2
    
    if df <= 0:
        raise ValueError(f"Número de classes ({n_classes}) muito pequeno. Deve ser > 3.")
        
    # Uso de sf (Survival Function) para maior precisão numérica
    p = stats.chi2.sf(estatistica, df)
    
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
    Retorna quantis teóricos e dados ordenados para a construção de Q-Q Plot.
    """
    x_sorted = np.sort(x)
    n = len(x)
    # Posições de plotagem de Blom/Hazen para os quantis
    p = (np.arange(1, n + 1) - 0.5) / n
    quantis_teoricos = stats.norm.ppf(p)
    return quantis_teoricos, x_sorted
