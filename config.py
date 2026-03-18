# config.py
from enum import Enum

class Config:
    APP_TITLE = "ENGESTAT"
    APP_SUBTITLE = "Ferramenta Estatística para Engenharia Civil"
    LAYOUT = "wide"
    
    # Mensagens de erro padronizadas
    class Mensagens:
        ERRO_MINIMO = "É necessário pelo menos {n} valores numéricos."
        ERRO_CONVERSAO = "Erro na conversão dos dados. Verifique o formato."
        ERRO_DIVISAO_ZERO = "Média zero impossibilita cálculo do CV."
    
    # Limiares para interpretações
    class Limiares:
        CV_BAIXO = 10
        CV_MODERADO = 20
        ASSIMETRIA_TOLERANCIA = 0.1

class OpcoesEntrada(Enum):
    MANUAL = "Inserção Manual"
    UPLOAD = "Upload de Planilha"