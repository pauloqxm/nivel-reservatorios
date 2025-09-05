import io
import re
import math
import unicodedata
from datetime import datetime, timezone
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Reservatórios – Tabela diária", layout="wide")

# ==========================
# Configuração e utilidades
# ==========================
SHEETS_URL = "https://docs.google.com/spreadsheets/d/1zZ0RCyYj-AzA_dhWzxRziDWjgforbaH7WIoSEd2EKdk/edit?gid=1305065127#gid=1305065127"

@st.cache_data(ttl=900)
def google_sheets_to_csv_url(url: str) -> str:
    """
    Converte URL de edição do Google Sheets para URL de exportação CSV
    Aceita formatos com /edit ou /view e com fragmento #gid=...
    """
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
    gid = None
    gid_match = re.search(r"[#?]gid=(\d+)", url)
    if gid_match:
        gid = gid_match.group(1)
    if not m:
        return url  # se não reconhecer, retorna como está (pode já ser CSV)
    doc_id = m.group(1)
    if gid is None:
        return f"https://docs.google.com/spreadsheets/d/{doc_id}/export?format=csv"
    return f"https://docs.google.com/spreadsheets/d/{doc_id}/export?format=csv&gid={gid}"

def strip_accents_lower(s: str) -> str:
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    return s.lower()

def to_number(x):
    """
    Converte strings como '1.234,56' -> 1234.56, preservando floats/ints/NaN.
    Remove espaços e caracteres não numéricos (exceto . , -)
    """
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return math.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return math.nan
    # mantém apenas dígitos, ponto, vírgula e sinal
    s = ''.join(ch for ch in s if ch.isdigit() or ch in ".,-")
    # se tiver vírgula como decimal, troca por ponto; remove pontos de milhar
    # heurística: se houver tanto ponto quanto vírgula, considere vírgula como decimal
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    else:
        # se só vírgula, vira decimal
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return math.nan

def to_datetime_any(x):
    """
    Converte datas em vários formatos (DD/MM/AAAA, AAAA-MM-DD, etc.)
    """
    if pd.isna(x):
        return pd.NaT
    # primeiro tenta pandas com dayfirst
    for dayfirst in (True, False):
        try:
            dt = pd.to_datetime(x, dayfirst=dayfirst, errors="raise")
            return dt
        except Exception:
            pass
    return pd.NaT

@st.cache_data(ttl=900)
def load_data_from_url(url: str) -> pd.DataFrame:
    csv_url = google_sheets_to_csv_url(url)
    df = pd.read_csv(csv_url, dtype=str)
    # tira espaços dos nomes
    df.columns = [c.strip() for c in df.columns]
    return df

def find_column(df: pd.DataFrame, aliases):
    """
    Encontra a primeira coluna em df cujos nomes normalizados batem com a lista de aliases.
    aliases: lista de strings (já sem acentos e em minúsculo)
    """
    normalized = {col: strip_accents_lower(col) for col in df.columns}
    for col, norm in normalized.items():
        if norm in aliases:
            return col
    # tentativa mais flexível: procura por alias contido
    for col, norm in normalized.items():
        if any(alias in norm for alias in aliases):
            return col
    return None

def compute_table(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # ====== Descobrir as colunas relevantes por nomes comuns ======
    # Ajuste aqui os aliases se sua planilha usar nomes diferentes
    col_reservatorio = find_column(df, {"reservatorio", "reservatório", "acude", "açude", "nome"})
    col_cota_sangria = find_column(df, {"cota sangria", "cota de sangria", "cota_sangria", "cota excedencia"})
    col_data = find_column(df, {"data", "dt", "dia"})
    col_volume = find_column(df, {"volume", "vol"})
    col_percentual = find_column(df, {"percentual", "perc", "percentual (%)", "volume (%)", "percentual (%)"})
    col_nivel = find_column(df, {"nivel", "nível", "cota", "altura"})

    required = {
        "Reservatório": col_reservatorio,
        "Cota Sangria": col_cota_sangria,
        "Data": col_data,
        "Volume": col_volume,
        "Percentual": col_percentual,
        "Nivel": col_nivel,
    }
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise ValueError(
            "Não foi possível identificar as colunas na planilha. "
            f"Faltando: {', '.join(missing)}.\n"
            "Dica: renomeie ou ajuste os aliases no código."
        )

    # ====== Conversões de tipos ======
    df[col_data] = df[col_data].apply(to_datetime_any)
    df[col_volume] = df[col_volume].apply(to_number)
    df[col_percentual] = df[col_percentual].apply(to_number)
    df[col_nivel] = df[col_nivel].apply(to_number)
    df[col_cota_sangria] = df[col_cota_sangria].apply(to_number)

    # mantém só linhas com data válida
    df = df.dropna(subset=[col_data])

    # ====== Seleciona registro atual e anterior por reservatório ======
    # atual = linha com maior data; anterior = maior data < data_atual
    dfs = []
    for res, dfr in df.groupby(col_reservatorio, dropna=True):
        dfr = dfr.sort_values(col_data)
        if dfr.empty:
            continue
        atual = dfr.iloc[-1]
        # anterior: última linha com data menor que a atual
        dfr_ant = dfr[dfr[col_data] < atual[col_data]]
        anterior = dfr_ant.iloc[-1] if not dfr_ant.empty else None

        # capacidade total = Volume / (Percentual/100) (usando dados da linha atual)
        vol = atual[col_volume]
        perc = atual[col_percentual]
        if perc and not math.isnan(perc) and perc != 0:
            capacidade_total = vol / (perc / 100.0) if vol and not math.isnan(vol) else math.nan
        else:
            capacidade_total = math.nan

        # variação do nível = nível_atual - nível_anterior
        nivel_atual = atual[col_nivel]
        nivel_ant = anterior[col_nivel] if anterior is not None else math.nan
        if (nivel_atual is None or math.isnan(nivel_atual)) or (nivel_ant is None or math.isnan(nivel_ant)):
            variacao_nivel = math.nan
        else:
            variacao_nivel = nivel_atual - nivel_ant

        # datas para exibição
        data_atual = atual[col_data]
        data_anterior = anterior[col_data] if anterior is not None else pd.NaT

        # cota de sangria (pega a da linha atual quando disponível)
        cota_sangria = atual[col_cota_sangria]

        dfs.append({
            "Reservatório": res,
            "Cota Sangria": cota_sangria,
            "Data Anterior": data_anterior.date() if pd.notna(data_anterior) else None,
            "Data Atual": data_atual.date() if pd.notna(data_atual) else None,
            "Capacidade Total (m³)": capacidade_total,
            "Variação do Nível": variacao_nivel,
        })

    out = pd.DataFrame(dfs)

    # Ordena por nome e coloca formatos bonitos
    if not out.empty:
        out = out.sort_values("Reservatório").reset_index(drop=True)

    return out

# ==========================
# UI
# ==========================
st.title("📊 Tabela diária de Reservatórios")

with st.sidebar:
    st.markdown("### Fonte dos dados")
    default_mode = st.radio(
        "Escolha a fonte",
        ["Google Sheets (link padrão)", "Enviar CSV (arquivo local)"],
        index=0,
    )
    if default_mode == "Google Sheets (link padrão)":
        url = st.text_input("URL do Google Sheets", value=SHEETS_URL)
        uploaded_file = None
    else:
        url = None
        uploaded_file = st.file_uploader("Envie um CSV com a mesma estrutura", type=["csv"])

# Carregamento
try:
    if uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file, dtype=str)
    else:
        df_raw = load_data_from_url(url)

    # Filtro opcional por reservatório (antes de computar, se desejar)
    col_res_guess = find_column(df_raw, {"reservatorio", "reservatório", "acude", "açude", "nome"})
    if col_res_guess:
        reservatorios = sorted(x for x in df_raw[col_res_guess].dropna().unique())
        sel = st.multiselect("Filtrar reservatórios (opcional)", reservatorios, [])
        df_filtered = df_raw[df_raw[col_res_guess].isin(sel)] if sel else df_raw
    else:
        df_filtered = df_raw

    result = compute_table(df_filtered)

    # Formatação e exibição
    st.subheader("Resultado")
    if result.empty:
        st.info("Nenhum dado encontrado com a estrutura esperada.")
    else:
        # Formata números
        result_fmt = result.copy()
        # Capacidade em m³ com separador de milhar (sem casas decimais se muito grande)
        result_fmt["Capacidade Total (m³)"] = result_fmt["Capacidade Total (m³)"].apply(
            lambda v: f"{v:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".") if pd.notna(v) else ""
        )
        # Variação com 2 casas decimais
        result_fmt["Variação do Nível"] = result_fmt["Variação do Nível"].apply(
            lambda v: f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") if pd.notna(v) else ""
        )
        # Cota de sangria com 2 casas
        result_fmt["Cota Sangria"] = result_fmt["Cota Sangria"].apply(
            lambda v: f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") if pd.notna(v) else ""
        )

        st.dataframe(
            result_fmt,
            use_container_width=True,
            hide_index=True
        )

        # Download do CSV bruto (sem formatação de string)
        csv_bytes = result.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Baixar CSV",
            data=csv_bytes,
            file_name="reservatorios_tabela_diaria.csv",
            mime="text/csv"
        )

        st.caption(
            "• **Capacidade Total (m³)** = Volume atual ÷ (Percentual atual ÷ 100). "
            "• **Variação do Nível** = Nível (Data Atual) − Nível (Data Anterior). "
            "• Quando não há registro no dia anterior, a variação fica em branco."
        )

except Exception as e:
    st.error(f"Ocorreu um erro ao processar os dados: {e}")
    st.stop()
