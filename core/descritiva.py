# core/descritiva.py
import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import List, Optional
from config import Config


@dataclass
class EstatisticasDescritivas:
    n: int
    media: float
    mediana: float
    moda: List[float]
    variancia: float
    desvio_padrao: float
    coeficiente_variacao: Optional[float]
    minimo: float
    maximo: float
    q1: float
    q3: float
    iqr: float
    assimetria: float
    curtose: float


def calcular_estatisticas(dados: List[float]) -> EstatisticasDescritivas:
    arr = np.array(dados, dtype=float)

    if len(arr) < 2:
        raise ValueError(Config.Mensagens.ERRO_MINIMO.format(n=2))

    media = float(np.mean(arr))
    mediana = float(np.median(arr))
    variancia = float(np.var(arr, ddof=1))
    desvio = float(np.std(arr, ddof=1))

    # Moda robusta (funciona para multimodal)
    valores, contagens = np.unique(arr, return_counts=True)
    max_freq = np.max(contagens)
    modas = valores[contagens == max_freq].tolist()

    # Coeficiente de variação protegido
    cv = (desvio / media) * 100 if media != 0 else None

    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1

    assimetria = float(stats.skew(arr)) if len(arr) > 2 else 0.0
    curtose = float(stats.kurtosis(arr)) if len(arr) > 3 else 0.0

    return EstatisticasDescritivas(
        n=len(arr),
        media=media,
        mediana=mediana,
        moda=modas,
        variancia=variancia,
        desvio_padrao=desvio,
        coeficiente_variacao=cv,
        minimo=float(np.min(arr)),
        maximo=float(np.max(arr)),
        q1=float(q1),
        q3=float(q3),
        iqr=float(iqr),
        assimetria=assimetria,
        curtose=curtose
    )