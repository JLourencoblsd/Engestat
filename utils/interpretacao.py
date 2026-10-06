# utils/interpretacao.py
from typing import Optional, Dict, Any, List
import pandas as pd
from core.descritiva import EstatisticasDescritivas

# ==============================================================================
# 1. ANÁLISE DESCRITIVA E EXPLORATÓRIA
# ==============================================================================

def interpretar_cv(cv: Optional[float]) -> str:
    """
    Interpreta o Coeficiente de Variação (CV), medindo a dispersão relativa dos dados.
    """
    if cv is None:
        return "⚠️ **Não aplicável:** A média é igual a zero, impossibilitando o cálculo da variabilidade relativa."
    
    # Normaliza caso o CV venha em escala decimal (0.15) ou percentual (15)
    cv_percent = cv if cv > 1 else cv * 100
    
    if cv_percent < 10:
        return f"🔵 **Baixa Dispersão / Homogêneo (CV = {cv_percent:.1f}%):** Os dados são muito parecidos entre si e estão bem concentrados ao redor da média. O processo demonstra alto grau de controle."
    elif cv_percent <= 20:
        return f"🟡 **Moderada Dispersão (CV = {cv_percent:.1f}%):** Existe uma oscilação aceitável dos valores. O processo apresenta estabilidade razoável sem desvios excessivos."
    else:
        return f"🔴 **Alta Dispersão / Heterogêneo (CV = {cv_percent:.1f}%):** Os dados oscilam bastante em relação à média. Indica alta instabilidade no processo ou grande variabilidade natural."


def interpretar_distribuicao(media: float, mediana: float) -> str:
    """
    Avalia a simetria da distribuição comparando Média e Mediana.
    """
    if media == 0:
        return "📊 A média é zero, impossibilitando a análise de diferença relativa."
    
    diferenca_relativa = abs(media - mediana) / abs(media)
    
    if diferenca_relativa < 0.05:
        return "⚖️ **Distribuição Simétrica:** Média e mediana possuem valores muito próximos. Os dados distribuem-se de forma equilibrada em torno do centro."
    elif media > mediana:
        return "📈 **Assimetria Positiva (à direita):** A média é maior que a mediana. Isso ocorre quando existem alguns valores extremamente altos (cauda longa à direita) que 'puxam' a média para cima."
    else:
        return "📉 **Assimetria Negativa (à esquerda):** A média é menor que a mediana. Isso ocorre quando existem alguns valores extremamente baixos (cauda longa à esquerda) que 'puxam' a média para baixo."


def interpretar_outliers(percentual_outliers: float) -> str:
    """
    Interpreta a presença e o impacto de valores discrepantes (outliers).
    """
    if percentual_outliers == 0:
        return "✅ **Sem Outliers:** Nenhuma observação discrepante foi detectada pelos critérios estatísticos."
    elif percentual_outliers < 5:
        return f"⚠️ **Presença Leve de Outliers ({percentual_outliers:.1f}% dos dados):** Foram identificados poucos pontos fora do padrão. Recomenda-se checar se foram erros de digitação/medição ou eventos raros legítimos."
    else:
        return f"🔴 **Elevada Presença de Outliers ({percentual_outliers:.1f}% dos dados):** Uma parcela significativa dos dados está fora do padrão esperável. Tais observações podem distorcer a média e o desvio-padrão."


def gerar_relatorio_completo(est: EstatisticasDescritivas, percentual_outliers: float) -> str:
    """
    Sintetiza as estatísticas descritivas em um laudo técnico e pedagógico.
    """
    cv_val = est.coeficiente_variacao if est.coeficiente_variacao is not None else 0.0
    cv_percent = cv_val if cv_val > 1 else cv_val * 100

    relatorio = f"""
### 📋 Laudo Descritivo da Amostra

* **Tamanho da Amostra ($n$):** {est.n} observações
* **Média (Ponto de Equilíbrio):** {est.media:.2f}
* **Mediana / Q2 (Ponto Central):** {est.mediana:.2f} (exactamente 50% dos dados estão abaixo e 50% acima deste valor)
* **Desvio-Padrão (Grau de Oscilação):** {est.desvio_padrao:.2f} (afastamento típico dos dados em relação à média)

**Mapeamento de Quantis:**
Mínimo ({est.minimo:.2f}) ➔ Q1/25% ({est.q1:.2f}) ➔ Mediana/50% ({est.mediana:.2f}) ➔ Q3/75% ({est.q3:.2f}) ➔ Máximo ({est.maximo:.2f})

**Diagnóstico Estatístico:**
• {interpretar_cv(est.coeficiente_variacao)}
• {interpretar_distribuicao(est.media, est.mediana)}
• {interpretar_outliers(percentual_outliers)}

**Recomendação Prática:**
{"✓ Os dados apresentam boa estabilidade. As métricas tradicionais (média e desvio-padrão) são representativas e confiáveis para modelagem." if cv_percent <= 20 and percentual_outliers < 5 else "⚠️ A amostra possui alta variabilidade ou presença de outliers. Recomenda-se utilizar a mediana e o intervalo interquartílico (IQR) como medidas centrais mais robustas, além de verificar a normalidade antes de aplicar testes paramétricos."}
    """
    return relatorio.strip()


# ==============================================================================
# 2. TESTES DE NORMALIDADE
# ==============================================================================

def interpretar_normalidade(pvalor: float, teste_nome: str = "") -> str:
    """
    Interpreta o p-valor de testes de normalidade (Shapiro-Wilk, Lilliefors, Qui-Quadrado).
    """
    nome_str = f" no teste {teste_nome}" if teste_nome else ""
    if pvalor > 0.05:
        return (
            f"🔵 **Hipótese de Normalidade Mantida (p = {pvalor:.4f}{nome_str}):**\n"
            f"Como o p-valor é maior que 0,05, **não há evidências estatísticas para rejeitar a normalidade**. "
            f"Podemos assumir que os dados seguem uma Distribuição Normal (curva em formato de sino)."
        )
    else:
        return (
            f"🔴 **Afastamento da Normalidade (p = {pvalor:.4f}{nome_str}):**\n"
            f"Como o p-valor é menor ou igual a 0,05, **rejeita-se a hipótese de normalidade**. "
            f"Existe evidência estatística de que os dados não seguem uma distribuição normal clássica."
        )


def recomendar_teste(p_normal: bool) -> str:
    """
    Recomenda as técnicas inferenciais adequadas com base na aderência à normalidade.
    """
    if p_normal:
        return (
            "✅ **Recomendação de Testes (Métodos Paramétricos):**\n"
            "Como o pressuposto de normalidade foi atendido, você pode utilizar métodos paramétricos de maior poder estatístico:\n"
            "• **Comparar 2 Grupos Independentes:** Teste t de Student (ou Teste t de Welch se as variâncias forem desiguais)\n"
            "• **Comparar 2 Grupos Pareados (Antes/Depois):** Teste t Pareado\n"
            "• **Comparar 3 ou mais Grupos:** Análise de Variância (ANOVA de 1 Fator)\n"
            "• **Modelagem Linear:** Regressão por Mínimos Quadrados Ordinários (MQO)"
        )
    else:
        return (
            "⚠️ **Recomendação de Testes (Métodos Não-Paramétricos):**\n"
            "Como os dados não seguem uma distribuição normal, recomenda-se usar métodos livres de distribuição (baseados em postos/ordenamento):\n"
            "• **Comparar 2 Grupos Independentes:** Teste U de Mann-Whitney\n"
            "• **Comparar 2 Grupos Pareados (Antes/Depois):** Teste de Wilcoxon\n"
            "• **Comparar 3 ou mais Grupos:** Teste de Kruskal-Wallis\n"
            "• **Alternativa:** Aplicar transformações de dados (ex: Logaritmo) antes de refazer o teste"
        )


# ==============================================================================
# 3. COMPARAÇÃO E AJUSTE DE DISTRIBUIÇÕES
# ==============================================================================

def interpretar_melhor_distribuicao(resultados: List[Dict[str, Any]], criterio: str = "AIC") -> str:
    """
    Indica qual distribuição teórica obteve o melhor ajuste aos dados.
    """
    if not resultados:
        return "Sem resultados para análise."
    
    melhor = min(resultados, key=lambda r: r[criterio])
    nome = melhor["Distribuição"]
    
    return (
        f"✅ **Distribuição de Melhor Ajuste:** A curva **{nome}** apresentou o melhor desempenho segundo o critério **{criterio}**.\n\n"
        f"*(Nota Metodológica: O {criterio} equilibra a precisão do ajuste com a simplicidade do modelo. Quanto **menor** esse valor, melhor o ajuste teórica e praticamente).* "
    )


def interpretar_comparacao_distribuicoes(resultados: List[Dict[str, Any]]) -> str:
    """
    Gera um comparativo detalhado das distribuições ajustadas usando AIC e BIC.
    """
    if not resultados:
        return "Sem dados disponíveis."

    linhas = ["**Resumo Comparativo de Modelos Probabilísticos:**"]
    for r in resultados:
        ad = f" (AD = {r['AD_Stat']:.4f})" if 'AD_Stat' in r and r['AD_Stat'] is not None else ""
        linhas.append(f"• **{r['Distribuição']}**: AIC = {r['AIC']:.2f} | BIC = {r['BIC']:.2f}{ad}")
    
    melhor_aic = min(resultados, key=lambda r: r["AIC"])
    melhor_bic = min(resultados, key=lambda r: r["BIC"])
    
    linhas.append("")
    linhas.append(f"🏆 **Vencedora pelo critério AIC (Akaike):** {melhor_aic['Distribuição']} (Valor = {melhor_aic['AIC']:.2f})")
    linhas.append(f"🏆 **Vencedora pelo critério BIC (Bayesiano):** {melhor_bic['Distribuição']} (Valor = {melhor_bic['BIC']:.2f})")
    
    if melhor_aic['Distribuição'] == melhor_bic['Distribuição']:
        linhas.append(f"\n💡 **Conclusão Unânime:** Ambos os critérios confirmam a distribuição **{melhor_aic['Distribuição']}** como o modelo ideal para seus dados.")
    else:
        linhas.append("\n💡 **Observação:** O critério BIC penaliza modelos com mais parâmetros de forma mais rigorosa. Em caso de divergência, adote a distribuição mais simples e fisicamente coerente com seu problema.")

    return "\n".join(linhas)


# ==============================================================================
# 4. INTERVALOS DE CONFIANÇA (ESTIMAÇÃO)
# ==============================================================================

def interpretar_ic_media(res: dict) -> str:
    conf = 100 * (1 - res['alpha'])
    return (
        f"**Intervalo de Confiança para a Média Populacional (μ)**\n"
        f"• Média Amostral ($\bar{{x}}$): {res['media']:.4f} (calculada para $n = {res['n']}$ observações)\n"
        f"• Nível de Confiança: **{conf:.0f}%**\n"
        f"• Intervalo Estimado: **[{res['li']:.4f} ; {res['ls']:.4f}]**\n\n"
        f"🎯 **O que significa:** Se repetirmos a coleta da amostra muitas vezes nas mesmas condições, "
        f"em **{conf:.0f}%** desses cenários a verdadeira média da população ($\mu$) estará dentro do intervalo **[{res['li']:.4f} ; {res['ls']:.4f}]**."
    )


def interpretar_ic_desvio(res: dict) -> str:
    conf = 100 * (1 - res['alpha'])
    texto = (
        f"**Intervalo de Confiança para o Desvio-Padrão Populacional (σ)**\n"
        f"• Desvio-Padrão Amostral ($s$): {res['S']:.4f}\n"
        f"• Nível de Confiança: **{conf:.0f}%**\n"
        f"• Intervalo Estimado: **[{res['li']:.4f} ; {res['ls']:.4f}]**\n\n"
        f"🎯 **O que significa:** Temos {conf:.0f}% de confiança de que a variabilidade real (desvio-padrão $\sigma$) de toda a população situa-se entre **{res['li']:.4f}** e **{res['ls']:.4f}**."
    )
    if 'aprox_normal' in res and res['aprox_normal']:
        an = res['aprox_normal']
        texto += f"\n\n*(Aproximação assintótica/normal para grandes amostras $n>30$: [{an['li']:.4f} ; {an['ls']:.4f}]).*"
    return texto


def interpretar_ic_variancia(res: dict) -> str:
    conf = 100 * (1 - res['alpha'])
    return (
        f"**Intervalo de Confiança para a Variância Populacional (σ²)**\n"
        f"• Variância Amostral ($s^2$): {res['S2']:.4f}\n"
        f"• Nível de Confiança: **{conf:.0f}%**\n"
        f"• Intervalo Estimado: **[{res['li']:.4f} ; {res['ls']:.4f}]**\n\n"
        f"🎯 **O que significa:** Temos {conf:.0f}% de confiança de que a variância real ($\sigma^2$) de todo o processo está contida nos limites **[{res['li']:.4f} ; {res['ls']:.4f}]**."
    )


# ==============================================================================
# 5. COMPARAÇÃO DE 2 AMOSTRAS
# ==============================================================================

def interpretar_teste_f(res_f: dict) -> str:
    p_val = res_f['p_valor']
    if res_f['rejeita_h0']:
        conc = "⚠️ **VARIÂNCIAS HETEROGÊNEAS (Heterocedasticidade):** Rejeita-se $H_0$. Há diferença estatisticamente significativa entre a oscilação/dispersão dos dois grupos."
    else:
        conc = "✅ **VARIÂNCIAS HOMOGÊNEAS (Homocedasticidade):** Não se rejeita $H_0$. As variâncias dos dois grupos são estatisticamente equivalentes."

    return (
        f"**Teste F de Homocedasticidade (Igualdade de Variâncias)**\n"
        f"• p-valor calculado: **{p_val:.4f}** (nível de significância α = 0,05)\n"
        f"• **Conclusão:** {conc}"
    )


def interpretar_teste_t(res_t: dict) -> str:
    p_val = res_t['p_valor']
    m1, m2 = res_t['media1'], res_t['media2']
    
    if res_t['rejeita_h0']:
        maior = "Grupo 1" if m1 > m2 else "Grupo 2"
        conc = (
            f"⚠️ **DIFERENÇA ESTATISTICAMENTE SIGNIFICATIVA:** Rejeita-se a hipótese nula ($H_0$). "
            f"A média do **{maior}** ({max(m1, m2):.2f}) é estatisticamente superior à outra (p = {p_val:.4f}). "
            f"A diferença observada não ocorreu por mero acaso."
        )
    else:
        conc = (
            f"⚖️ **DIFERENÇA NÃO SIGNIFICATIVA (Empate Técnico):** Não se rejeita a hipótese nula ($H_0$). "
            f"Embora haja uma pequena variação entre as médias amostrais ({m1:.2f} vs {m2:.2f}), ela é atribuível à flutuação amostragem aleatória (p = {p_val:.4f} > 0,05)."
        )

    return (
        f"**Teste t de Student para Comparação de Médias**\n"
        f"• p-valor calculado: **{p_val:.4f}**\n"
        f"• **Conclusão:** {conc}"
    )


# ==============================================================================
# 6. ANOVA E COMPARAÇÕES MÚLTIPLAS (TUKEY)
# ==============================================================================

def interpretar_anova(res_a: dict) -> str:
    p_val = res_a['p_valor']
    if res_a['rejeita_h0']:
        conc = (
            f"⚠️ **Pelo menos um grupo difere dos demais:** Rejeita-se a hipótese nula ($H_0$). "
            f"Existe evidência estatística de que ao menos uma das médias populacionais é diferente das outras (p = {p_val:.4f}). "
            f"Recomenda-se verificar o Teste de Tukey abaixo para descobrir exatamente quais grupos diferem entre si."
        )
    else:
        conc = (
            f"⚖️ **Sem diferenças significativas entre os tratamentos:** Não se rejeita $H_0$. "
            f"Todas as médias comparadas são estatisticamente equivalentes ao nível de 5% de significância (p = {p_val:.4f})."
        )

    return (
        f"**Análise de Variância (ANOVA de 1 Fator)**\n"
        f"• p-valor (Estatística F): **{p_val:.4f}**\n"
        f"• **Conclusão:** {conc}"
    )


def interpretar_tukey(df_tukey: pd.DataFrame) -> str:
    linhas = []
    for _, row in df_tukey.iterrows():
        # Compatibilidade com colunas personalizadas ou padrões do statsmodels
        sig = row.get('Diferença Significativa?', False)
        p_val = row.get('P-valor', row.get('p-adj', 1.0))
        dif = row.get('Diferença', row.get('meandiff', 0.0))
        g_a = row.get('Grupo A', row.get('group1', 'A'))
        g_b = row.get('Grupo B', row.get('group2', 'B'))

        if sig or p_val <= 0.05:
            linhas.append(f"🥊 **{g_a} vs {g_b}**: Diferença média de {dif:.3f} (Diferença real comprovada, p-valor ajustado = {p_val:.4f})")

    if not linhas:
        return (
            "**Teste Post-Hoc de Tukey (Comparações Par a Par):**\n"
            "Após comparar grupo por grupo com ajuste de probabilidade, nenhuma diferença foi conclusiva. Todos empatam estatisticamente."
        )

    return (
        "**Teste Post-Hoc de Tukey (Identificação das Diferenças):**\n" +
        "\n".join(linhas)
    )
