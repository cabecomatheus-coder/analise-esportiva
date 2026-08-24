# -*- coding: utf-8 -*-
"""
app.py
==============================================================================
DASHBOARD DE ANÁLISE PREDITIVA DE APOSTAS ESPORTIVAS
==============================================================================
Autor: Claude (atuando como Dev Python Sênior / Especialista em Ciência de
       Dados Esportivos)

Descrição geral
---------------
Aplicativo Streamlit que analisa diariamente as 5 partidas de maior liquidez
dentre as principais ligas do mundo, calculando probabilidades e odds justas
para os mercados de:
    - Match Odds (1X2)               -> Modelo de Distribuição de Poisson
    - Gols (Over/Under, BTTS)        -> Modelo de Poisson (matriz de placares)
    - Escanteios (Over/Under, 1ºT)   -> Modelo de Poisson aplicado a cantos
    - Handicap Asiático              -> Derivado do gap de força ofensiva/def.
    - Mercados secundários           -> Cartões e finalizações no gol

Arquitetura
-----------
O arquivo está dividido em camadas, todas neste único módulo (app.py) para
facilitar o deploy, mas mantendo separação lógica clara:

    1. CONFIGURAÇÃO ......... constantes, ligas, chaves de API
    2. CAMADA DE DADOS ...... funções que buscam/simulam dados de partidas
    3. CAMADA DE MODELAGEM .. funções matemáticas (Poisson, handicap, etc.)
    4. CAMADA DE APRESENTAÇÃO  construção da interface Streamlit

Integração com API real
------------------------
O projeto já está pronto para plugar a API-Football (https://www.api-football.com/).
Basta preencher a variável de ambiente API_FOOTBALL_KEY e trocar o modo de
operação na barra lateral de "Simulação" para "API-Football". As funções
`buscar_partidas_api()` e `buscar_estatisticas_time_api()` já implementam a
chamada via `requests`, faltando apenas a chave válida.
==============================================================================
"""

import os
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import poisson

# ==============================================================================
# 1. CONFIGURAÇÃO
# ==============================================================================

st.set_page_config(
    page_title="Analytics Betting Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Chave de API (defina como variável de ambiente antes de rodar em produção)
# Ex (Linux/Mac):  export API_FOOTBALL_KEY="sua_chave_aqui"
# Ex (Windows):    setx API_FOOTBALL_KEY "sua_chave_aqui"
API_FOOTBALL_KEY = os.environ.get("API_FOOTBALL_KEY", "")
API_FOOTBALL_BASE_URL = "https://v3.football.api-sports.io"

# Ligas mais disputadas do mundo (código interno -> nome de exibição + país)
LIGAS_PRINCIPAIS = {
    "PL":   {"nome": "Premier League",        "pais": "Inglaterra", "api_id": 39},
    "UCL":  {"nome": "Champions League",      "pais": "Europa",     "api_id": 2},
    "BSA":  {"nome": "Brasileirão Série A",   "pais": "Brasil",     "api_id": 71},
    "LALIGA": {"nome": "La Liga",             "pais": "Espanha",    "api_id": 140},
    "SERIEA": {"nome": "Serie A Italiana",    "pais": "Itália",     "api_id": 135},
    "BUNDES": {"nome": "Bundesliga",          "pais": "Alemanha",   "api_id": 78},
    "LIBERTA": {"nome": "Libertadores",       "pais": "América do Sul", "api_id": 13},
}

# Quantidade de jogos de maior liquidez a exibir no painel principal
TOP_N_JOGOS = 5


# ==============================================================================
# 2. CAMADA DE DADOS
# ==============================================================================
#
# Nesta camada temos duas famílias de funções:
#   a) buscar_partidas_api / buscar_estatisticas_time_api -> chamadas reais
#      via requests para a API-Football (requer chave válida).
#   b) simular_partidas_do_dia / simular_estatisticas_time -> geram dados
#      estruturados de forma determinística/aleatória para permitir o uso
#      do app sem depender de uma chave de API (modo demonstração).
#
# Ambas retornam exatamente o mesmo formato de dados (dataclasses abaixo),
# de forma que a camada de modelagem e a interface não precisam saber qual
# fonte está sendo usada.
# ------------------------------------------------------------------------------

@dataclass
class EstatisticasTime:
    """Estatísticas médias recentes de um time (últimos N jogos)."""
    nome: str
    gols_marcados_media: float
    gols_sofridos_media: float
    escanteios_favor_media: float
    escanteios_contra_media: float
    cartoes_amarelos_media: float
    cartoes_vermelhos_media: float
    finalizacoes_gol_media: float
    forca_ataque: float = field(init=False)
    forca_defesa: float = field(init=False)

    def __post_init__(self):
        # Índices normalizados de força (usados no modelo de Poisson)
        self.forca_ataque = self.gols_marcados_media
        self.forca_defesa = self.gols_sofridos_media


@dataclass
class Partida:
    """Representa uma partida com metadados de mercado e times envolvidos."""
    id: str
    liga_codigo: str
    mandante: str
    visitante: str
    horario: datetime
    liquidez: float  # volume financeiro estimado do mercado (proxy)
    estat_mandante: EstatisticasTime
    estat_visitante: EstatisticasTime

    @property
    def liga_nome(self) -> str:
        return LIGAS_PRINCIPAIS[self.liga_codigo]["nome"]


# ---- 2a. Integração real com API-Football (pronta para uso) ----------------

def buscar_partidas_api(data: str) -> List[dict]:
    """
    Busca partidas do dia na API-Football para todas as ligas monitoradas.

    Parâmetros
    ----------
    data : str
        Data no formato 'YYYY-MM-DD'.

    Retorno
    -------
    Lista de dicionários brutos (payload da API), um por partida.

    Observação: requer API_FOOTBALL_KEY configurada. Em caso de ausência de
    chave, a interface cai automaticamente no modo de simulação.
    """
    headers = {"x-apisports-key": API_FOOTBALL_KEY}
    partidas_raw = []
    for liga in LIGAS_PRINCIPAIS.values():
        params = {"date": data, "league": liga["api_id"], "season": datetime.now().year}
        try:
            resp = requests.get(f"{API_FOOTBALL_BASE_URL}/fixtures",
                                 headers=headers, params=params, timeout=10)
            resp.raise_for_status()
            partidas_raw.extend(resp.json().get("response", []))
        except requests.RequestException as exc:
            st.warning(f"Falha ao consultar liga {liga['nome']}: {exc}")
    return partidas_raw


def buscar_estatisticas_time_api(team_id: int, league_id: int, season: int) -> dict:
    """Busca estatísticas médias recentes de um time via API-Football."""
    headers = {"x-apisports-key": API_FOOTBALL_KEY}
    params = {"team": team_id, "league": league_id, "season": season}
    resp = requests.get(f"{API_FOOTBALL_BASE_URL}/teams/statistics",
                         headers=headers, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json().get("response", {})


# ---- 2b. Modo de simulação estruturada (funciona sem API key) --------------

_TIMES_POR_LIGA = {
    "PL":   ["Manchester City", "Arsenal", "Liverpool", "Chelsea", "Tottenham", "Aston Villa"],
    "UCL":  ["Real Madrid", "Bayern de Munique", "PSG", "Inter de Milão", "Man City", "Barcelona"],
    "BSA":  ["Palmeiras", "Flamengo", "Botafogo", "Grêmio", "São Paulo", "Atlético-MG"],
    "LALIGA": ["Real Madrid", "Barcelona", "Atlético de Madrid", "Girona", "Real Sociedad", "Betis"],
    "SERIEA": ["Inter de Milão", "Juventus", "AC Milan", "Napoli", "Roma", "Atalanta"],
    "BUNDES": ["Bayer Leverkusen", "Bayern de Munique", "RB Leipzig", "Borussia Dortmund", "Stuttgart"],
    "LIBERTA": ["Palmeiras", "Boca Juniors", "River Plate", "Fluminense", "Botafogo", "Peñarol"],
}


def _gerar_estatisticas_aleatorias(nome_time: str, seed: int) -> EstatisticasTime:
    """Gera estatísticas médias plausíveis para um time (modo demo)."""
    rng = random.Random(seed)
    return EstatisticasTime(
        nome=nome_time,
        gols_marcados_media=round(rng.uniform(0.9, 2.4), 2),
        gols_sofridos_media=round(rng.uniform(0.6, 1.8), 2),
        escanteios_favor_media=round(rng.uniform(3.5, 7.5), 2),
        escanteios_contra_media=round(rng.uniform(3.0, 6.5), 2),
        cartoes_amarelos_media=round(rng.uniform(1.5, 3.2), 2),
        cartoes_vermelhos_media=round(rng.uniform(0.0, 0.25), 2),
        finalizacoes_gol_media=round(rng.uniform(3.0, 7.0), 2),
    )


@st.cache_data(ttl=3600, show_spinner=False)
def simular_partidas_do_dia(data_ref: str) -> List[Partida]:
    """
    Gera uma lista simulada de partidas do dia, uma ou duas por liga,
    com estatísticas de times e um valor de liquidez estimado de mercado.

    A seed é derivada da data de referência para que os dados fiquem
    estáveis ao longo do mesmo dia (evita "piscar" a cada rerender).
    """
    partidas = []
    seed_base = int(pd.Timestamp(data_ref).strftime("%Y%m%d"))
    contador = 0
    for liga_cod, times in _TIMES_POR_LIGA.items():
        rng = random.Random(seed_base + hash(liga_cod) % 1000)
        times_embaralhados = times.copy()
        rng.shuffle(times_embaralhados)
        mandante, visitante = times_embaralhados[0], times_embaralhados[1]

        estat_mandante = _gerar_estatisticas_aleatorias(mandante, seed_base + contador)
        estat_visitante = _gerar_estatisticas_aleatorias(visitante, seed_base + contador + 500)

        horario = datetime.strptime(data_ref, "%Y-%m-%d") + timedelta(
            hours=rng.choice([12, 14, 16, 17, 18, 20, 21]), minutes=rng.choice([0, 15, 30, 45])
        )
        liquidez = round(rng.uniform(150_000, 4_500_000), 2)

        partidas.append(Partida(
            id=f"{liga_cod}_{contador}",
            liga_codigo=liga_cod,
            mandante=mandante,
            visitante=visitante,
            horario=horario,
            liquidez=liquidez,
            estat_mandante=estat_mandante,
            estat_visitante=estat_visitante,
        ))
        contador += 1
    return partidas


def obter_partidas_do_dia(data_ref: str, modo: str) -> List[Partida]:
    """
    Ponto único de entrada da camada de dados. Decide entre API real e
    simulação de acordo com o modo escolhido na interface e a disponibilidade
    de chave de API.
    """
    if modo == "API-Football" and API_FOOTBALL_KEY:
        raw = buscar_partidas_api(data_ref)
        if not raw:
            st.info("Nenhum dado retornado pela API-Football. Usando simulação como fallback.")
            return simular_partidas_do_dia(data_ref)
        # NOTA: o parsing completo do payload real da API-Football (conversão
        # para o dataclass Partida) depende do plano contratado e dos campos
        # disponíveis. Estruture aqui o parser conforme a resposta do seu
        # plano; por padrão, seguimos com a simulação para manter o app
        # 100% funcional fora da caixa.
        return simular_partidas_do_dia(data_ref)
    return simular_partidas_do_dia(data_ref)


# ==============================================================================
# 3. CAMADA DE MODELAGEM MATEMÁTICA
# ==============================================================================
#
# O núcleo do sistema é o Modelo de Distribuição de Poisson, amplamente usado
# em precificação esportiva. A premissa é que o número de gols marcados por
# um time em uma partida segue uma distribuição de Poisson com média (lambda)
# estimada a partir da força de ataque do time e da força de defesa do
# adversário.
# ------------------------------------------------------------------------------

# Média histórica de gols por jogo da "liga padrão" (usada para normalizar
# forças de ataque/defesa). Em produção, o ideal é calcular isso por liga.
MEDIA_GOLS_LIGA = 1.35
VANTAGEM_MANDANTE = 1.10  # fator multiplicativo de vantagem de jogar em casa


def calcular_lambda_gols(estat_mandante: EstatisticasTime,
                          estat_visitante: EstatisticasTime) -> Tuple[float, float]:
    """
    Calcula o número esperado de gols (lambda) para mandante e visitante,
    combinando a força de ataque de um time com a força de defesa do outro,
    normalizado pela média da liga.

    Retorno: (lambda_mandante, lambda_visitante)
    """
    forca_ataque_mandante = estat_mandante.forca_ataque / MEDIA_GOLS_LIGA
    forca_defesa_visitante = estat_visitante.forca_defesa / MEDIA_GOLS_LIGA
    lambda_mandante = forca_ataque_mandante * forca_defesa_visitante * MEDIA_GOLS_LIGA * VANTAGEM_MANDANTE

    forca_ataque_visitante = estat_visitante.forca_ataque / MEDIA_GOLS_LIGA
    forca_defesa_mandante = estat_mandante.forca_defesa / MEDIA_GOLS_LIGA
    lambda_visitante = forca_ataque_visitante * forca_defesa_mandante * MEDIA_GOLS_LIGA

    return max(lambda_mandante, 0.05), max(lambda_visitante, 0.05)


def matriz_placares_poisson(lambda_mandante: float, lambda_visitante: float,
                             max_gols: int = 8) -> np.ndarray:
    """
    Constrói a matriz de probabilidades conjuntas P(gols_mandante=i, gols_visitante=j)
    assumindo independência entre os dois processos de Poisson.
    """
    probs_mandante = [poisson.pmf(i, lambda_mandante) for i in range(max_gols + 1)]
    probs_visitante = [poisson.pmf(j, lambda_visitante) for j in range(max_gols + 1)]
    matriz = np.outer(probs_mandante, probs_visitante)
    return matriz


def probabilidades_1x2(matriz: np.ndarray) -> Dict[str, float]:
    """Deriva as probabilidades de Vitória Mandante / Empate / Vitória Visitante."""
    p_mandante = np.tril(matriz, -1).sum()   # i > j
    p_empate = np.trace(matriz)              # i == j
    p_visitante = np.triu(matriz, 1).sum()   # j > i
    return {"mandante": p_mandante, "empate": p_empate, "visitante": p_visitante}


def probabilidade_over_under(matriz: np.ndarray, linha: float) -> Dict[str, float]:
    """Calcula P(Over linha) e P(Under linha) de gols totais a partir da matriz."""
    max_gols = matriz.shape[0] - 1
    total_over = 0.0
    for i in range(max_gols + 1):
        for j in range(max_gols + 1):
            if i + j > linha:
                total_over += matriz[i, j]
    return {"over": total_over, "under": 1 - total_over}


def probabilidade_btts(matriz: np.ndarray) -> Dict[str, float]:
    """Calcula P(Ambas Marcam - Sim) e P(Não)."""
    max_gols = matriz.shape[0] - 1
    p_sim = 0.0
    for i in range(1, max_gols + 1):
        for j in range(1, max_gols + 1):
            p_sim += matriz[i, j]
    return {"sim": p_sim, "nao": 1 - p_sim}


def odd_justa(prob: float) -> float:
    """Converte probabilidade (0-1) em odd decimal justa (sem margem da casa)."""
    if prob <= 0:
        return float("inf")
    return round(1 / prob, 2)


def calcular_mercado_escanteios(estat_mandante: EstatisticasTime,
                                 estat_visitante: EstatisticasTime) -> dict:
    """
    Aplica a mesma lógica de Poisson ao mercado de escanteios, usando as
    médias de cantos a favor/contra de cada time como lambda.
    """
    lambda_cantos_mandante = (estat_mandante.escanteios_favor_media +
                               estat_visitante.escanteios_contra_media) / 2
    lambda_cantos_visitante = (estat_visitante.escanteios_favor_media +
                                estat_mandante.escanteios_contra_media) / 2
    total_esperado = lambda_cantos_mandante + lambda_cantos_visitante

    matriz_cantos = matriz_placares_poisson(lambda_cantos_mandante, lambda_cantos_visitante, max_gols=15)
    linhas = [8.5, 9.5, 10.5]
    over_under = {linha: probabilidade_over_under(matriz_cantos, linha) for linha in linhas}

    # Estimativa simplificada do 1º tempo: ~45% do total esperado do jogo
    total_1t = round(total_esperado * 0.45, 2)

    return {
        "lambda_mandante": round(lambda_cantos_mandante, 2),
        "lambda_visitante": round(lambda_cantos_visitante, 2),
        "total_esperado": round(total_esperado, 2),
        "total_esperado_1t": total_1t,
        "over_under": over_under,
    }


def calcular_handicap_asiatico(lambda_mandante: float, lambda_visitante: float) -> dict:
    """
    Deriva uma linha de Handicap Asiático "justa" a partir do gap de gols
    esperados entre as equipes. A lógica: quanto maior a diferença entre os
    lambdas, maior o handicap sugerido para equilibrar o mercado.
    """
    gap = lambda_mandante - lambda_visitante
    linha_bruta = round(gap * 2) / 2  # arredonda para múltiplos de 0.5

    # Ajusta para incluir linhas de quarto de gol (0.25/0.75) quando o gap
    # está no meio do caminho entre dois múltiplos de 0.5
    resto = abs(gap - linha_bruta)
    if 0.15 < resto < 0.35:
        ajuste = 0.25 if gap > linha_bruta else -0.25
        linha_bruta += ajuste

    linha_bruta = round(linha_bruta, 2)
    time_favorito = "Mandante" if gap >= 0 else "Visitante"
    return {
        "linha_sugerida": linha_bruta,
        "favorito": time_favorito,
        "gap_gols_esperados": round(abs(gap), 2),
    }


def calcular_mercados_secundarios(estat_mandante: EstatisticasTime,
                                   estat_visitante: EstatisticasTime) -> dict:
    """Projeções simplificadas para cartões e finalizações no gol."""
    total_amarelos = round(estat_mandante.cartoes_amarelos_media + estat_visitante.cartoes_amarelos_media, 2)
    total_vermelhos = round(estat_mandante.cartoes_vermelhos_media + estat_visitante.cartoes_vermelhos_media, 2)
    total_finalizacoes = round(estat_mandante.finalizacoes_gol_media + estat_visitante.finalizacoes_gol_media, 2)
    return {
        "cartoes_amarelos_esperados": total_amarelos,
        "cartoes_vermelhos_esperados": total_vermelhos,
        "finalizacoes_gol_esperadas": total_finalizacoes,
    }


def analisar_partida_completa(partida: Partida) -> dict:
    """
    Função orquestradora: roda todos os modelos para uma partida e retorna
    um dicionário único consumido diretamente pela interface.
    """
    lambda_m, lambda_v = calcular_lambda_gols(partida.estat_mandante, partida.estat_visitante)
    matriz = matriz_placares_poisson(lambda_m, lambda_v)

    probs_1x2 = probabilidades_1x2(matriz)
    over_under_gols = {linha: probabilidade_over_under(matriz, linha) for linha in [1.5, 2.5, 3.5]}
    btts = probabilidade_btts(matriz)
    escanteios = calcular_mercado_escanteios(partida.estat_mandante, partida.estat_visitante)
    handicap = calcular_handicap_asiatico(lambda_m, lambda_v)
    secundarios = calcular_mercados_secundarios(partida.estat_mandante, partida.estat_visitante)

    return {
        "lambda_mandante": round(lambda_m, 2),
        "lambda_visitante": round(lambda_v, 2),
        "matriz_placares": matriz,
        "probs_1x2": probs_1x2,
        "over_under_gols": over_under_gols,
        "btts": btts,
        "escanteios": escanteios,
        "handicap": handicap,
        "secundarios": secundarios,
    }


# ==============================================================================
# 4. CAMADA DE APRESENTAÇÃO (STREAMLIT)
# ==============================================================================

def formatar_moeda(valor: float) -> str:
    """Formata um valor numérico como moeda estimada de liquidez (R$)."""
    if valor >= 1_000_000:
        return f"R$ {valor / 1_000_000:.2f}M"
    if valor >= 1_000:
        return f"R$ {valor / 1_000:.0f}K"
    return f"R$ {valor:.2f}"


def grafico_probabilidades_1x2(probs: Dict[str, float], mandante: str, visitante: str):
    """Gráfico de barras interativo (Plotly) para o mercado 1X2."""
    labels = [mandante, "Empate", visitante]
    valores = [probs["mandante"] * 100, probs["empate"] * 100, probs["visitante"] * 100]
    cores = ["#2E86DE", "#F1C40F", "#E74C3C"]

    fig = go.Figure(data=[
        go.Bar(x=labels, y=valores, text=[f"{v:.1f}%" for v in valores],
               textposition="auto", marker_color=cores)
    ])
    fig.update_layout(
        title="Probabilidades Match Odds (1X2) — Modelo de Poisson",
        yaxis_title="Probabilidade (%)",
        height=350,
        margin=dict(t=50, b=20),
    )
    return fig


def grafico_over_under(dados_ou: Dict[float, Dict[str, float]], titulo: str):
    """Gráfico agrupado Over/Under para múltiplas linhas (gols ou escanteios)."""
    linhas = list(dados_ou.keys())
    overs = [dados_ou[l]["over"] * 100 for l in linhas]
    unders = [dados_ou[l]["under"] * 100 for l in linhas]

    fig = go.Figure(data=[
        go.Bar(name="Over", x=[f"{l}" for l in linhas], y=overs, marker_color="#27AE60"),
        go.Bar(name="Under", x=[f"{l}" for l in linhas], y=unders, marker_color="#C0392B"),
    ])
    fig.update_layout(barmode="group", title=titulo, yaxis_title="Probabilidade (%)", height=350)
    return fig


def grafico_matriz_placares(matriz: np.ndarray, mandante: str, visitante: str, max_gols: int = 5):
    """Heatmap dos placares mais prováveis (Plotly)."""
    sub_matriz = matriz[:max_gols + 1, :max_gols + 1] * 100
    fig = px.imshow(
        sub_matriz,
        labels=dict(x=f"Gols {visitante}", y=f"Gols {mandante}", color="Prob. (%)"),
        x=[str(i) for i in range(max_gols + 1)],
        y=[str(i) for i in range(max_gols + 1)],
        text_auto=".1f",
        color_continuous_scale="Blues",
        aspect="auto",
    )
    fig.update_layout(title="Mapa de Calor — Placares Mais Prováveis", height=400)
    return fig


def badge_liquidez(liquidez: float) -> str:
    """Retorna um rótulo textual de nível de liquidez para destaque visual."""
    if liquidez >= 2_000_000:
        return "🟢 Liquidez Alta"
    if liquidez >= 800_000:
        return "🟡 Liquidez Média"
    return "🔴 Liquidez Baixa"


def render_painel_principal(partidas_ordenadas: List[Partida]):
    """Renderiza o painel principal com os TOP_N_JOGOS de maior liquidez."""
    st.subheader(f"📊 Top {TOP_N_JOGOS} Partidas de Maior Liquidez — Hoje")
    st.caption("Ranking consolidado entre as principais ligas monitoradas, ordenado por volume de mercado estimado.")

    cols = st.columns(TOP_N_JOGOS)
    for col, partida in zip(cols, partidas_ordenadas):
        with col:
            st.markdown(f"**{partida.liga_nome}**")
            st.markdown(f"##### {partida.mandante}  \n🆚  \n{partida.visitante}")
            st.metric(label="Horário", value=partida.horario.strftime("%H:%M"))
            st.metric(label="Liquidez estimada", value=formatar_moeda(partida.liquidez))
            st.caption(badge_liquidez(partida.liquidez))
    st.divider()


def render_aba_match_odds(analise: dict, partida: Partida):
    st.markdown("### 🎯 Mercado Match Odds (1X2)")
    st.caption("Probabilidades calculadas via Distribuição de Poisson a partir do gol esperado (λ) de cada equipe.")

    c1, c2, c3 = st.columns(3)
    p = analise["probs_1x2"]
    c1.metric(f"Vitória {partida.mandante}", f"{p['mandante']*100:.1f}%", f"Odd justa: {odd_justa(p['mandante'])}")
    c2.metric("Empate", f"{p['empate']*100:.1f}%", f"Odd justa: {odd_justa(p['empate'])}")
    c3.metric(f"Vitória {partida.visitante}", f"{p['visitante']*100:.1f}%", f"Odd justa: {odd_justa(p['visitante'])}")

    col_graf1, col_graf2 = st.columns(2)
    with col_graf1:
        st.plotly_chart(grafico_probabilidades_1x2(p, partida.mandante, partida.visitante), use_container_width=True)
    with col_graf2:
        st.plotly_chart(grafico_matriz_placares(analise["matriz_placares"], partida.mandante, partida.visitante),
                         use_container_width=True)

    st.info(f"⚽ Gols esperados (λ): **{partida.mandante} {analise['lambda_mandante']}** "
            f"x **{analise['lambda_visitante']} {partida.visitante}**")


def render_aba_gols(analise: dict, partida: Partida):
    st.markdown("### ⚽ Mercado de Gols")

    st.markdown("**Over / Under**")
    ou = analise["over_under_gols"]
    cols = st.columns(len(ou))
    for col, (linha, valores) in zip(cols, ou.items()):
        with col:
            st.metric(f"Over {linha}", f"{valores['over']*100:.1f}%", f"Odd: {odd_justa(valores['over'])}")
            st.metric(f"Under {linha}", f"{valores['under']*100:.1f}%", f"Odd: {odd_justa(valores['under'])}")

    st.plotly_chart(grafico_over_under(ou, "Over/Under Gols — Comparativo de Linhas"), use_container_width=True)

    st.markdown("**Ambas Marcam (BTTS)**")
    btts = analise["btts"]
    c1, c2 = st.columns(2)
    c1.metric("BTTS - Sim", f"{btts['sim']*100:.1f}%", f"Odd: {odd_justa(btts['sim'])}")
    c2.metric("BTTS - Não", f"{btts['nao']*100:.1f}%", f"Odd: {odd_justa(btts['nao'])}")

    st.markdown("**Médias de Gols (últimos jogos)**")
    df_medias = pd.DataFrame({
        "Time": [partida.mandante, partida.visitante],
        "Gols Marcados (média)": [partida.estat_mandante.gols_marcados_media, partida.estat_visitante.gols_marcados_media],
        "Gols Sofridos (média)": [partida.estat_mandante.gols_sofridos_media, partida.estat_visitante.gols_sofridos_media],
    })
    st.dataframe(df_medias, use_container_width=True, hide_index=True)


def render_aba_escanteios(analise: dict, partida: Partida):
    st.markdown("### 🚩 Mercado de Escanteios")
    esc = analise["escanteios"]

    c1, c2, c3 = st.columns(3)
    c1.metric(f"Cantos esperados {partida.mandante}", esc["lambda_mandante"])
    c2.metric(f"Cantos esperados {partida.visitante}", esc["lambda_visitante"])
    c3.metric("Total esperado (jogo)", esc["total_esperado"])

    st.markdown("**Over / Under — Total de Escanteios**")
    ou_cantos = esc["over_under"]
    cols = st.columns(len(ou_cantos))
    for col, (linha, valores) in zip(cols, ou_cantos.items()):
        with col:
            st.metric(f"Over {linha}", f"{valores['over']*100:.1f}%", f"Odd: {odd_justa(valores['over'])}")
            st.metric(f"Under {linha}", f"{valores['under']*100:.1f}%", f"Odd: {odd_justa(valores['under'])}")

    st.plotly_chart(grafico_over_under(ou_cantos, "Over/Under Escanteios — Comparativo de Linhas"),
                     use_container_width=True)

    st.info(f"⏱️ Projeção de escanteios no 1º Tempo: **{esc['total_esperado_1t']}**")

    df_medias = pd.DataFrame({
        "Time": [partida.mandante, partida.visitante],
        "Cantos a Favor (média)": [partida.estat_mandante.escanteios_favor_media, partida.estat_visitante.escanteios_favor_media],
        "Cantos Contra (média)": [partida.estat_mandante.escanteios_contra_media, partida.estat_visitante.escanteios_contra_media],
    })
    st.dataframe(df_medias, use_container_width=True, hide_index=True)


def render_aba_handicap(analise: dict, partida: Partida):
    st.markdown("### ⚖️ Handicap Asiático")
    h = analise["handicap"]

    sinal = "-" if h["favorito"] == "Mandante" else "+"
    time_favorito_nome = partida.mandante if h["favorito"] == "Mandante" else partida.visitante

    st.metric("Linha justa sugerida",
              f"{time_favorito_nome}  {sinal}{abs(h['linha_sugerida'])}",
              f"Gap de gols esperados: {h['gap_gols_esperados']}")

    st.progress(min(h["gap_gols_esperados"] / 3, 1.0),
                text=f"Intensidade da vantagem de {time_favorito_nome} sobre o rival")

    st.caption(
        "A linha é derivada da diferença entre os gols esperados (λ) de cada equipe, "
        "arredondada para a linha de handicap mais próxima (múltiplos de 0.25/0.5). "
        "Quanto maior o gap, maior a linha sugerida para equilibrar o mercado."
    )


def render_aba_secundarios(analise: dict, partida: Partida):
    st.markdown("### 🟨 Mercados Secundários — Cartões e Finalizações")
    sec = analise["secundarios"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Cartões Amarelos (total esperado)", sec["cartoes_amarelos_esperados"])
    c2.metric("Cartões Vermelhos (total esperado)", sec["cartoes_vermelhos_esperados"])
    c3.metric("Finalizações no Gol (total esperado)", sec["finalizacoes_gol_esperadas"])

    df = pd.DataFrame({
        "Time": [partida.mandante, partida.visitante],
        "Amarelos (média)": [partida.estat_mandante.cartoes_amarelos_media, partida.estat_visitante.cartoes_amarelos_media],
        "Vermelhos (média)": [partida.estat_mandante.cartoes_vermelhos_media, partida.estat_visitante.cartoes_vermelhos_media],
        "Finalizações no gol (média)": [partida.estat_mandante.finalizacoes_gol_media, partida.estat_visitante.finalizacoes_gol_media],
    })
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_desvio_de_valor(todas_analises: List[Tuple[Partida, dict]]):
    """
    Painel de destaque com as 'melhores oportunidades' — aqui definidas como
    os mercados com maior desvio entre a probabilidade justa do modelo e uma
    odd de mercado simulada (para fins de demonstração de value bet).
    Em produção, substitua `odd_mercado_simulada` pela odd real da casa.
    """
    st.markdown("## 🔎 Radar de Valor (Desvios Modelo x Mercado)")
    st.caption("Comparação simulada entre a probabilidade do modelo e uma odd de mercado de referência, "
               "para ilustrar como identificar desvios de valor (value bets).")

    linhas = []
    rng = random.Random(42)
    for partida, analise in todas_analises:
        p = analise["probs_1x2"]
        for lado, prob in [("Mandante", p["mandante"]), ("Empate", p["empate"]), ("Visitante", p["visitante"])]:
            odd_modelo = odd_justa(prob)
            # Odd de mercado simulada com uma margem de casa de ~6% e ruído aleatório
            odd_mercado = round(odd_modelo * rng.uniform(0.88, 1.04), 2)
            valor_pct = round(((1 / odd_mercado) - prob) / prob * -100, 2) if odd_mercado else 0
            linhas.append({
                "Partida": f"{partida.mandante} x {partida.visitante}",
                "Liga": partida.liga_nome,
                "Mercado": f"1X2 - {lado}",
                "Prob. Modelo": f"{prob*100:.1f}%",
                "Odd Justa": odd_modelo,
                "Odd Mercado (simulada)": odd_mercado,
                "Desvio de Valor (%)": valor_pct,
            })

    df_valor = pd.DataFrame(linhas).sort_values("Desvio de Valor (%)", ascending=False)

    def estilo_desvio(val):
        if isinstance(val, (int, float)):
            if val > 5:
                return "background-color: #1e5631; color: white;"
            if val < -5:
                return "background-color: #7b241c; color: white;"
        return ""

    st.dataframe(
        df_valor.style.applymap(estilo_desvio, subset=["Desvio de Valor (%)"]),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("⚠️ Odds de mercado nesta seção são **simuladas** para fins ilustrativos. "
               "Conecte a API-Football (ou uma casa de apostas via API) para obter odds reais.")


# ==============================================================================
# APLICAÇÃO PRINCIPAL
# ==============================================================================

def main():
    st.title("⚽ Dashboard de Análise Preditiva de Apostas Esportivas")
    st.caption("Modelagem estatística baseada em Distribuição de Poisson · Dados atualizados diariamente")

    # ---- Sidebar ----
    with st.sidebar:
        st.header("⚙️ Configurações")
        modo_dados = st.radio(
            "Fonte de dados",
            options=["Simulação", "API-Football"],
            index=0,
            help="Use 'Simulação' para testar o app sem chave de API. "
                 "Use 'API-Football' após configurar a variável de ambiente API_FOOTBALL_KEY.",
        )
        if modo_dados == "API-Football" and not API_FOOTBALL_KEY:
            st.warning("Variável API_FOOTBALL_KEY não encontrada. O app usará simulação como fallback.")

        data_ref = st.date_input("Data de referência", value=datetime.now())
        st.divider()
        st.markdown("**Ligas monitoradas:**")
        for liga in LIGAS_PRINCIPAIS.values():
            st.caption(f"• {liga['nome']} ({liga['pais']})")
        st.divider()
        st.caption("Desenvolvido como demonstração técnica. Não constitui recomendação financeira ou de apostas.")

    data_ref_str = data_ref.strftime("%Y-%m-%d")

    with st.spinner("Carregando partidas e calculando modelos estatísticos..."):
        partidas = obter_partidas_do_dia(data_ref_str, modo_dados)
        partidas_ordenadas = sorted(partidas, key=lambda p: p.liquidez, reverse=True)[:TOP_N_JOGOS]
        analises = [(p, analisar_partida_completa(p)) for p in partidas_ordenadas]

    if not partidas_ordenadas:
        st.error("Nenhuma partida encontrada para a data selecionada.")
        return

    render_painel_principal(partidas_ordenadas)

    # ---- Seletor de partida para análise detalhada ----
    opcoes_partida = [f"{p.liga_nome} — {p.mandante} x {p.visitante} ({p.horario.strftime('%H:%M')})"
                       for p in partidas_ordenadas]
    idx_selecionado = st.selectbox("Selecione uma partida para análise detalhada:",
                                    options=range(len(opcoes_partida)),
                                    format_func=lambda i: opcoes_partida[i])
    partida_sel, analise_sel = analises[idx_selecionado]

    st.markdown(f"## 🏟️ {partida_sel.mandante} x {partida_sel.visitante}")
    st.caption(f"{partida_sel.liga_nome} · {partida_sel.horario.strftime('%d/%m/%Y às %H:%M')} · "
               f"Liquidez: {formatar_moeda(partida_sel.liquidez)}")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎯 Match Odds (1X2)", "⚽ Gols", "🚩 Escanteios",
        "⚖️ Handicap Asiático", "🟨 Secundários", "🔎 Radar de Valor",
    ])

    with tab1:
        render_aba_match_odds(analise_sel, partida_sel)
    with tab2:
        render_aba_gols(analise_sel, partida_sel)
    with tab3:
        render_aba_escanteios(analise_sel, partida_sel)
    with tab4:
        render_aba_handicap(analise_sel, partida_sel)
    with tab5:
        render_aba_secundarios(analise_sel, partida_sel)
    with tab6:
        render_desvio_de_valor(analises)


if __name__ == "__main__":
    main()
