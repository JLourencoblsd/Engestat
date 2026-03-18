# core/distribuicoes.py
import numpy as np
from scipy import stats
from typing import List, Dict, Any, Tuple

# Distribuições suportadas (nome: objeto scipy.stats)
DISTRIBUICOES = {
    "Normal": stats.norm,
    "Gamma": stats.gamma,
    "Lognormal": stats.lognorm,
    "Weibull": stats.weibull_min,
    "Inverse Gaussian": stats.invgauss
}

def ajustar_distribuicoes(dados: List[float]) -> List[Dict[str, Any]]:
    """
    Ajusta várias distribuições aos dados e retorna métricas comparativas.
    
    Args:
        dados: lista de valores numéricos
        
    Returns:
        lista de dicionários, cada um com:
        - Distribuição (str)
        - Parâmetros (tuple)
        - AIC (float)
        - BIC (float)
        - KS_p (float)
        - AD_Stat (float ou None)
    """
    x = np.array(dados)
    resultados = []
    
    for nome, dist in DISTRIBUICOES.items():
        # Ajuste da distribuição
        params = dist.fit(x)
        
        # Log-verossimilhança (evitar log(0))
        pdf_vals = dist.pdf(x, *params)
        logL = np.sum(np.log(pdf_vals + 1e-12))
        
        k = len(params)
        n = len(x)
        
        # Critérios de informação
        aic = 2 * k - 2 * logL
        bic = k * np.log(n) - 2 * logL
        
        # Teste Kolmogorov-Smirnov
        ks_stat, ks_p = stats.kstest(x, dist.cdf, args=params)
        
        # Teste Anderson-Darling (apenas para distribuições suportadas)
        ad_stat = None
        # Mapeamento de nomes amigáveis para os nomes exigidos por scipy.stats.anderson
        mapa_ad = {
            "Normal": 'norm',
            "Weibull": 'weibull_min',  # Anderson-Darling aceita 'weibull_min'?
            # Outras distribuições podem não ser suportadas
        }
        if nome in mapa_ad and mapa_ad[nome] in ['norm', 'expon', 'logistic', 'gumbel']:
            try:
                ad_result = stats.anderson(x, dist=mapa_ad[nome])
                ad_stat = ad_result.statistic
            except:
                pass  # se falhar, mantém None
        
        resultados.append({
            "Distribuição": nome,
            "Parâmetros": params,
            "AIC": aic,
            "BIC": bic,
            "KS_p": ks_p,
            "AD_Stat": ad_stat
        })
    
    return resultados

def melhor_distribuicao(resultados: List[Dict[str, Any]], criterio: str = "AIC") -> str:
    """
    Retorna o nome da distribuição com melhor (menor) valor do critério.
    """
    return min(resultados, key=lambda r: r[criterio])["Distribuição"]

def dados_para_pdf(dados: List[float], resultados: List[Dict[str, Any]]) -> Dict:
    """
    Prepara dados para plotar PDFs sobrepostas ao histograma.
    Retorna um dicionário com:
    - 'x_vals': eixo x comum
    - 'hist': dados para histograma
    - 'pdfs': lista de {nome, y_vals}
    """
    x = np.array(dados)
    x_vals = np.linspace(min(x), max(x), 200)
    hist_counts, bin_edges = np.histogram(x, bins='auto', density=True)
    
    pdfs = []
    for r in resultados:
        nome = r["Distribuição"]
        dist = DISTRIBUICOES[nome]
        y = dist.pdf(x_vals, *r["Parâmetros"])
        pdfs.append({"nome": nome, "y": y.tolist()})
    
    return {
        "x_vals": x_vals.tolist(),
        "hist": {"edges": bin_edges.tolist(), "densities": hist_counts.tolist()},
        "pdfs": pdfs
    }

def dados_para_qqplot(dados: List[float], resultados: List[Dict[str, Any]]) -> List[Dict]:
    """
    Prepara dados para Q-Q plots de cada distribuição.
    Retorna lista de {nome, teorico, observado}
    """
    x = np.array(dados)
    x_sorted = np.sort(x)
    n = len(x)
    # probabilidades teóricas (posições de plotagem)
    p = (np.arange(1, n+1) - 0.5) / n
    
    qq_data = []
    for r in resultados:
        nome = r["Distribuição"]
        dist = DISTRIBUICOES[nome]
        quantis_teoricos = dist.ppf(p, *r["Parâmetros"])
        qq_data.append({
            "nome": nome,
            "teorico": quantis_teoricos.tolist(),
            "observado": x_sorted.tolist()
        })
    return qq_data

def dados_para_cdf(dados: List[float], resultados: List[Dict[str, Any]]) -> List[Dict]:
    """
    Prepara dados para CDF empírica vs teórica.
    Retorna lista de {nome, x, cdf_emp, cdf_teo}
    """
    x = np.array(dados)
    x_sorted = np.sort(x)
    n = len(x)
    cdf_emp = np.arange(1, n+1) / (n+1)  # (i)/(n+1) evita 0 e 1 extremos
    
    cdf_data = []
    for r in resultados:
        nome = r["Distribuição"]
        dist = DISTRIBUICOES[nome]
        cdf_teo = dist.cdf(x_sorted, *r["Parâmetros"])
        cdf_data.append({
            "nome": nome,
            "x": x_sorted.tolist(),
            "cdf_emp": cdf_emp.tolist(),
            "cdf_teo": cdf_teo.tolist()
        })
    return cdf_data