# utils/interpretacao.py
from typing import Optional
from core.descritiva import EstatisticasDescritivas
import pandas as pd

# ============================================
# Funções existentes (mantidas)
# ============================================

def interpretar_cv(cv: Optional[float]) -> str:
    """Interpreta coeficiente de variação com recomendação técnica"""
    if cv is None:
        return "⚠️ A média é zero, então não conseguimos calcular o nível de variação com essa métrica."
    
    if cv < 10:
        return "🔵 **Dados super consistentes (Baixa Variação):** Seus dados são muito parecidos entre si. O processo parece bem controlado."
    elif cv < 20:
        return "🟡 **Variação aceitável (Moderada):** Existem algumas diferenças entre os valores, mas nada muito extremo. É bom ficar de olho."
    else:
        return "🔴 **Muita bagunça (Alta Variação):** Seus dados estão oscilando demais! O processo é instável e pouco previsível."

def interpretar_distribuicao(media: float, mediana: float) -> str:
    """Analisa simetria da distribuição"""
    if media == 0:
        return "📊 A média é zero, não é possível avaliar a balança dos dados."
    
    diferenca_relativa = abs(media - mediana) / abs(media)
    
    if diferenca_relativa < 0.1:
        return "⚖️ **Bem equilibrado (Simétrico):** A maioria dos dados está no meio, sem pender muito nem para valores altos, nem para baixos."
    elif media > mediana:
        return "📈 **Puxado para cima:** A maioria dos resultados é baixa, mas existem alguns valores tão altos que estão 'puxando' sua média para cima."
    else:
        return "📉 **Puxado para baixo:** A maioria dos resultados é alta, mas existem alguns valores tão baixos que estão 'puxando' sua média para baixo."

def interpretar_outliers(percentual_outliers: float) -> str:
    """Interpreta presença de outliers"""
    if percentual_outliers == 0:
        return "✅ **Tudo nos conformes:** Não achamos nenhum 'ponto fora da curva' (outlier)."
    elif percentual_outliers < 5:
        return f"⚠️ **Atenção ({percentual_outliers:.1f}% de dados estranhos):** Encontramos alguns valores muito diferentes do resto. Vale a pena checar se foi erro de medição ou se são reais."
    else:
        return f"🔴 **Alerta Vermelho ({percentual_outliers:.1f}% de dados estranhos):** Tem muita coisa fora do padrão! É altamente recomendável revisar como esses dados foram coletados."

def gerar_relatorio_completo(est: EstatisticasDescritivas, percentual_outliers: float) -> str:
    """Gera relatório técnico completo a partir do objeto EstatisticasDescritivas"""
    relatorio = f"""
### 📋 Laudo dos seus Dados

**Quantos dados temos:** {est.n} observações (linhas)  
**Onde está o meio:** Em média, o valor é {est.media:.2f} (mas o valor exato do meio é {est.mediana:.2f})  
**O tamanho da oscilação:** Em média, os dados fogem {est.desvio_padrao:.2f} para mais ou para menos.  

**Raio-X (Mínimo ao Máximo):** Começa em {est.minimo:.2f} ➔ Passa por {est.q1:.2f} (25%) ➔ Meio em {est.mediana:.2f} (50%) ➔ Chega a {est.q3:.2f} (75%) ➔ Termina em {est.maximo:.2f}  

**O que o software concluiu:**
• {interpretar_cv(est.coeficiente_variacao)}
• {interpretar_distribuicao(est.media, est.mediana)}
• {interpretar_outliers(percentual_outliers)}

**Dica de ouro do App:** {"Os dados estão bem comportados! Você pode usar ferramentas estatísticas tradicionais sem medo." if est.coeficiente_variacao and est.coeficiente_variacao < 20 else "Os dados estão muito instáveis. Talvez você precise 'limpar' pontos fora da curva ou usar testes especiais (não-paramétricos)."}
    """
    return relatorio

# ============================================
# Funções para Testes de Normalidade
# ============================================

def interpretar_normalidade(pvalor: float, teste_nome: str = "") -> str:
    """
    Interpreta o p-valor de um teste de normalidade.
    
    Args:
        pvalor: p-valor obtido no teste
        teste_nome: nome do teste (opcional, para personalizar a mensagem)
    
    Returns:
        String com interpretação
    """
    if pvalor > 0.05:
        return f"🔵 **Dados com comportamento previsível (p={pvalor:.4f}):** Eles seguem a famosa Curva Normal (formato de sino). Isso é ótimo, facilita muito as próximas análises!"
    else:
        return f"🔴 **Dados sem padrão normal (p={pvalor:.4f}):** Eles não formam o tradicional 'Sino'. Não tem problema, mas teremos que usar ferramentas específicas para lidar com eles."

def recomendar_teste(p_normal: bool) -> str:
    """
    Recomenda testes paramétricos ou não paramétricos com base na normalidade.
    
    Args:
        p_normal: True se os dados são normais, False caso contrário.
    
    Returns:
        String com recomendação.
    """
    if p_normal:
        return (
            "✅ **Próximos passos sugeridos:** Como seus dados são 'normais', você pode usar as vias expressas da estatística:\n"
            "- Comparar 2 grupos? Use o **Teste t** (Teste t de Student)\n"
            "- Comparar 3 ou mais grupos? Use a **ANOVA**\n"
            "- Calcular margens de segurança usando a distribuição t"
        )
    else:
        return (
            "⚠️ **Próximos passos sugeridos:** Como seus dados fugiram do padrão normal, precisaremos usar métodos 'todo-terreno':\n"
            "- Comparar 2 grupos? Use o **Teste de Mann-Whitney**\n"
            "- Comparar a mesma peça antes e depois? Use o **Teste de Wilcoxon**\n"
            "- Comparar vários grupos? Use o **Teste de Kruskal-Wallis**"
        )

# ============================================
# Funções para Comparação de Distribuições
# ============================================

def interpretar_melhor_distribuicao(resultados, criterio="AIC") -> str:
    """
    Retorna texto interpretativo sobre a melhor distribuição ajustada.
    
    Args:
        resultados: lista de dicionários retornada por ajustar_distribuicoes
        criterio: 'AIC' ou 'BIC'
    
    Returns:
        String com interpretação.
    """
    if not resultados:
        return "Sem resultados para análise."
    
    melhor = min(resultados, key=lambda r: r[criterio])
    nome = melhor["Distribuição"]
    
    texto = f"✅ **O molde perfeito:** De todas as curvas que o app testou, a que melhor 'veste' os seus dados é a distribuição **{nome}**.\n\n"
    texto += f"*(O critério usado foi o {criterio}. Quanto menor esse valor, melhor o ajuste matematicamente falando).* "
    return texto

def interpretar_comparacao_distribuicoes(resultados) -> str:
    """
    Gera um resumo comparativo simples das distribuições ajustadas.
    
    Args:
        resultados: lista de dicionários retornada por ajustar_distribuicoes
    
    Returns:
        String com tabela textual e observações.
    """
    linhas = ["**Resumo dos testes de formato:**"]
    for r in resultados:
        ad = f"{r['AD_Stat']:.4f}" if r['AD_Stat'] else "N/A"
        linhas.append(
            f"- **{r['Distribuição']}**: Margens de erro (AIC={r['AIC']:.2f}, BIC={r['BIC']:.2f})"
        )
    
    # Identifica a melhor por AIC e BIC
    melhor_aic = min(resultados, key=lambda r: r["AIC"])
    melhor_bic = min(resultados, key=lambda r: r["BIC"])
    
    linhas.append("")
    linhas.append(f"🏆 **Vencedora pelo critério AIC:** {melhor_aic['Distribuição']}")
    linhas.append(f"🏆 **Vencedora pelo critério BIC:** {melhor_bic['Distribuição']}")
    
    return "\n".join(linhas)

# ============================================
# Funções para Estimação
# ============================================

def interpretar_ic_media(res: dict) -> str:
    return (f"**Margem de Segurança da {res['parametro']}** \n"
            f"A média amostral que você calculou é {res['media']:.3f}.  \n"
            f"🎯 **O que isso significa:** Levando em conta sua amostra (n={res['n']}), o software garante com {100*(1-res['alpha']):.0f}% de certeza que a verdadeira média de toda a população do seu problema está entre **[{res['li']:.3f} e {res['ls']:.3f}]**.")

def interpretar_ic_desvio(res: dict) -> str:
    texto = (f"**Margem de Segurança do {res['parametro']} (Oscilação)** \n"
             f"O desvio amostral encontrado foi {res['S']:.3f}.  \n"
             f"🎯 **O que isso significa:** Temos {100*(1-res['alpha']):.0f}% de certeza que a variação real do processo inteiro está entre **[{res['li']:.3f} e {res['ls']:.3f}]**.")
    if 'aprox_normal' in res:
        an = res['aprox_normal']
        texto += f"\n\n**Como você tem muitos dados (n>30), o intervalo ajustado é:** [{an['li']:.3f} e {an['ls']:.3f}]"
    return texto

def interpretar_ic_variancia(res: dict) -> str:
    return (f"**Margem de Segurança da {res['parametro']}** \n"
            f"A variância da sua amostra é {res['S2']:.3f}.  \n"
            f"🎯 **O que isso significa:** Temos {100*(1-res['alpha']):.0f}% de certeza que a variância real do problema todo está entre **[{res['li']:.3f} e {res['ls']:.3f}]**.")

# ============================================
# Funções para Comparação de 2 Amostras
# ============================================

def interpretar_teste_f(res_f: dict) -> str:
    conc = "✅ **IGUAIS** (As duas amostras oscilam com a mesma intensidade)" if not res_f['rejeita_h0'] else "⚠️ **DIFERENTES** (Uma amostra varia muito mais que a outra)"
    return (f"**Comparação do nível de 'bagunça' (Teste F de Variâncias)** \n"
            f"Tirando a prova real (p-valor = {res_f['p_valor']:.4f}):  \n"
            f"Conclusão: Para a estatística, as variâncias são {conc}.")

def interpretar_teste_t(res_t: dict) -> str:
    if res_t['rejeita_h0']:
        if res_t['media1'] > res_t['media2']:
            quem = "**Grupo 1** é estatisticamente maior que o Grupo 2"
        else:
            quem = "**Grupo 2** é estatisticamente maior que o Grupo 1"
        conc = f"A diferença é REAL e não foi por acaso. {quem}."
    else:
        conc = "⚖️ **Empate Técnico!** As médias podem parecer diferentes no papel, mas para a estatística a diferença é obra do acaso. Considere-as iguais."
    
    return (f"**Duelo de Médias (Teste t)** \n"
            f"Tirando a prova real (p-valor = {res_t['p_valor']:.4f}):  \n"
            f"Conclusão: {conc}")

# ============================================
# Funções para ANOVA e Tukey
# ============================================

def interpretar_anova(res_a: dict) -> str:
    if res_a['rejeita_h0']:
        conc = "⚠️ **Tem alguém diferente na sala!** O teste provou que pelo menos um desses grupos não está empatando com o resto."
    else:
        conc = "⚖️ **Empate Geral!** Todos os grupos tiveram resultados tão parecidos que a estatística os considera iguais."
    
    return (f"**Comparação de Vários Grupos (ANOVA)** \n"
            f"Tirando a prova real (p-valor = {res_a['p_valor']:.4f}):  \n"
            f"Conclusão: {conc}")

def interpretar_tukey(df_tukey: pd.DataFrame) -> str:
    linhas = []
    for _, row in df_tukey.iterrows():
        if row['Diferença Significativa?']:
            linhas.append(f"🥊 {row['Grupo A']} vs {row['Grupo B']}: A diferença de {row['Diferença']:.3f} é **REAL** (p-valor = {row['P-valor']:.4f})")
    if not linhas:
        return "Após olhar par por par, nenhuma diferença foi conclusiva o suficiente. Todos empatam."
    return "**Quem é diferente de quem? (Tira-teima de Tukey):** \n" + "\n".join(linhas)