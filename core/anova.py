# core/anova.py
import numpy as np
import pandas as pd
from scipy.stats import f
from typing import Dict, Any, List, Tuple
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def anova_um_fator(grupos: Dict[str, List[float]], alpha: float = 0.05) -> Dict[str, Any]:
    """
    Executa ANOVA para um fator (delineamento inteiramente casualizado).
    
    Args:
        grupos: dicionário {nome_do_grupo: lista de valores}
        alpha: nível de significância
    
    Returns:
        dicionário com todas as somas de quadrados, GL, QM, F, e tabela
    """
    tratamentos = list(grupos.keys())
    k = len(tratamentos)
    
    # Preparar dados
    todos_dados = []
    totais = []
    ns = []
    medias = []
    variancias = []
    
    for nome in tratamentos:
        arr = np.array(grupos[nome], dtype=float)
        todos_dados.extend(arr)
        totais.append(arr.sum())
        ns.append(len(arr))
        medias.append(arr.mean())
        variancias.append(arr.var(ddof=1))
    
    N = len(todos_dados)
    G = sum(todos_dados)
    C = G**2 / N
    
    # Somas de quadrados
    sq_total = sum(x**2 for x in todos_dados) - C
    sq_trat = sum(t**2 / n for t, n in zip(totais, ns)) - C
    sq_res = sq_total - sq_trat
    
    # Graus de liberdade
    gl_trat = k - 1
    gl_res = N - k
    gl_total = N - 1
    
    # Quadrados médios
    qm_trat = sq_trat / gl_trat
    qm_res = sq_res / gl_res
    
    # Estatística F
    f_calc = qm_trat / qm_res
    f_critico = f.ppf(1 - alpha, gl_trat, gl_res)
    p_valor = 1 - f.cdf(f_calc, gl_trat, gl_res)
    
    # Tabela ANOVA (formato texto para exibição)
    tabela = pd.DataFrame({
        "Fonte": ["Tratamentos", "Resíduo", "Total"],
        "GL": [gl_trat, gl_res, gl_total],
        "SQ": [sq_trat, sq_res, sq_total],
        "QM": [qm_trat, qm_res, None],
        "F": [f_calc, None, None],
        "p-valor": [p_valor, None, None]
    })
    
    return {
        "grupos": tratamentos,
        "ns": ns,
        "medias": dict(zip(tratamentos, medias)),
        "totais": dict(zip(tratamentos, totais)),
        "sq_trat": sq_trat,
        "sq_res": sq_res,
        "sq_total": sq_total,
        "gl_trat": gl_trat,
        "gl_res": gl_res,
        "gl_total": gl_total,
        "qm_trat": qm_trat,
        "qm_res": qm_res,
        "f_calc": f_calc,
        "f_critico": f_critico,
        "p_valor": p_valor,
        "rejeita_h0": f_calc > f_critico,
        "tabela": tabela,
        "alpha": alpha
    }

def tukey_hsd(grupos: Dict[str, List[float]], alpha: float = 0.05) -> pd.DataFrame:
    """
    Realiza o teste de Tukey HSD para comparações múltiplas.
    
    Returns:
        DataFrame com resultados (pares, diferença, p-ajustado, rejeição)
    """
    # Preparar dados no formato longo
    valores = []
    rotulos = []
    for nome, dados in grupos.items():
        valores.extend(dados)
        rotulos.extend([nome] * len(dados))
    
    tukey = pairwise_tukeyhsd(endog=valores, groups=rotulos, alpha=alpha)
    
    # Converter para DataFrame e renomear colunas
    df = pd.DataFrame(data=tukey.summary().data[1:],
                      columns=tukey.summary().data[0])
    df = df.rename(columns={
        'group1': 'Grupo A',
        'group2': 'Grupo B',
        'meandiff': 'Diferença',
        'p-adj': 'P-valor',
        'lower': 'Lim. Inf.',
        'upper': 'Lim. Sup.',
        'reject': 'Diferença Significativa?'
    })
    return df