import io
import re
import math
import unicodedata
from datetime import datetime
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Reservatórios – Tabela diária", layout="wide")

# ==========================
# Configuração
# ==========================
SHEETS_URL = "https://docs.google.com/spreadsheets/d/1zZ0RCyYj-AzA_dhWzxRziDWjgforbaH7WIoSEd2EKdk/edit?gid=1305065127#gid=1305065127"

# ==========================
# Utilitários
# ==========================
@st.cache_data(ttl=900)
def google_sheets_to_csv_url(url: str) -> str:
    """
    Converte URL de edição/visualização do Google Sheets para URL de exportação CSV.
    """
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
    gid = None
    gid_match = re.search(r"[#?]gid=(\d+)", url)
    if gid_match:
        gid = gid_match.group(1)
    if not m:
        return url  # já pode ser um CSV direto
    doc_id = m.group(1)
    if gid is None:
        return f"https://docs.google.com/spreadsheets/d/{doc_id}/export?format=csv"
    return f"https://docs.google.com/spreadsheets/d/{doc_id}/export?format=csv&gid={gid}"

def strip_accents_lower(s: str) -> str:
    """Remove acentos de uma string e a converte para minúsculas."""
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    return s.lower()

def to_number(x):
    """
    Converte strings para números, lidando com diferentes formatos (ex: '1.234,56').
    A lógica agora é mais robusta para distinguir separadores de milhar e decimal.
    """
    if x is None or (isinstance(x, float) and math.isnan(x)) or isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().replace(" ", "")
    if s == "" or s.lower() in {"nan", "none"}:
        return math.nan
    
    # Heurística para separadores: o último é o decimal
    last_comma_pos = s.rfind(',')
    last_dot_pos = s.rfind('.')
    
    # Se ambos os separadores existem e a vírgula é a última
    if last_comma_pos > last_dot_pos:
        s = s.replace(".", "").replace(",", ".")
    # Se ambos os separadores existem e o ponto é o último
    elif last_dot_pos > last_comma_pos:
        s = s.replace(",", "")
    # Se só há vírgula, assume que é decimal
    elif last_comma_pos != -1:
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
    for dayfirst in (True, False):
        try:
            return pd.to_datetime(x, dayfirst=dayfirst, errors="raise")
        except Exception:
            pass
    return pd.NaT

@st.cache_data(ttl=900)
def load_data_from_url(url: str) -> pd.DataFrame:
    csv_url = google_sheets_to_csv_url(url)
    df = pd.read_csv(csv_url, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    return df

def find_column(df: pd.DataFrame, aliases):
    """
    Encontra a primeira coluna do DF cujo nome (sem acento e minúsculo) bate com aliases.
    Aceita correspondência exata ou 'contém'.
    """
    if isinstance(aliases, set):
        aliases = list(aliases)
    aliases = set(aliases)
    normalized = {col: strip_accents_lower(col) for col in df.columns}
    # Exata
    for col, norm in normalized.items():
        if norm in aliases:
            return col
    # Contém
    for col, norm in normalized.items():
        if any(alias in norm for alias in aliases):
            return col
    return None

# ==========================
# Cálculo principal (datas globais)
# ==========================
def compute_table_global_dates(df_raw: pd.DataFrame) -> pd.DataFrame:
    import numpy as np

    def last_scalar_on_date(dfr: pd.DataFrame, date_col: str, target_date, value_col: str):
        """
        Retorna o último valor (float) para a data exata target_date; NaN se não existir.
        Garante sempre um escalar (evita Series ambíguo).
        """
        if pd.isna(target_date):
            return math.nan
        sel = dfr.loc[dfr[date_col] == target_date, value_col]
        if sel.empty:
            return math.nan
        # força escalar
        try:
            return float(pd.to_numeric(sel, errors="coerce").dropna().iloc[-1])
        except Exception:
            return math.nan

    df = df_raw.copy()

    # Mapear colunas por nomes comuns/aliases
    col_reservatorio = find_column(df, {"reservatorio", "reservatório", "acude", "açude", "nome"})
    col_cota_sangria = find_column(df, {"cota sangria", "cota de sangria", "cota_sangria", "cota excedencia"})
    col_data           = find_column(df, {"data", "dt", "dia"})
    col_volume       = find_column(df, {"volume", "vol"})
    col_percentual  = find_column(df, {"percentual", "perc", "percentual (%)", "volume (%)"})
    col_nivel        = find_column(df, {"nivel", "nível", "cota", "altura"})

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
            "Dica: renomeie as colunas na planilha ou ajuste os aliases no código."
        )

    # Conversões
    df[col_data]           = df[col_data].apply(to_datetime_any)
    df[col_volume]         = df[col_volume].apply(to_number)
    df[col_percentual]     = df[col_percentual].apply(to_number)
    df[col_nivel]          = df[col_nivel].apply(to_number)
    df[col_cota_sangria]   = df[col_cota_sangria].apply(to_number)
    df = df.dropna(subset=[col_data])

    # Duas datas globais mais recentes
    unique_dates = pd.Series(df[col_data].dropna().unique()).sort_values().tolist()
    if len(unique_dates) == 0:
        return pd.DataFrame()
    data_atual     = pd.to_datetime(unique_dates[-1])
    data_anterior = pd.to_datetime(unique_dates[-2]) if len(unique_dates) >= 2 else pd.NaT

    # Rótulos de coluna (dd/mm/aaaa)
    col_atual_label    = data_atual.strftime("%d/%m/%Y") if pd.notna(data_atual) else "Data Atual"
    col_anterior_label = data_anterior.strftime("%d/%m/%Y") if pd.notna(data_anterior) else "Data Anterior"

    rows = []
    for res, dfr in df.groupby(col_reservatorio, dropna=True):
        # NÍVEL nas duas datas (preenche valores das colunas rotuladas pelas datas)
        nivel_atual    = last_scalar_on_date(dfr, col_data, data_atual,     col_nivel)
        nivel_anterior = last_scalar_on_date(dfr, col_data, data_anterior, col_nivel) if pd.notna(data_anterior) else math.nan

        # Volume e Percentual do DIA ATUAL (para capacidade total)
        vol_atual  = last_scalar_on_date(dfr, col_data, data_atual, col_volume)
        perc_atual = last_scalar_on_date(dfr, col_data, data_atual, col_percentual)

        if pd.notna(perc_atual) and perc_atual != 0 and pd.notna(vol_atual):
            cap_total = vol_atual / (perc_atual / 100.0)
        else:
            cap_total = math.nan

        # Cota de sangria (preferir valor no dia atual; senão, último não-nulo do histórico)
        cota_atual = last_scalar_on_date(dfr, col_data, data_atual, col_cota_sangria)
        if pd.notna(cota_atual):
            cota_sangria_val = cota_atual
        else:
            cota_hist = pd.to_numeric(dfr[col_cota_sangria], errors="coerce").dropna()
            cota_sangria_val = float(cota_hist.iloc[-1]) if not cota_hist.empty else math.nan

        # Variação (Nível atual - Nível anterior)
        variacao = (nivel_atual - nivel_anterior) if (pd.notna(nivel_atual) and pd.notna(nivel_anterior)) else math.nan

        rows.append({
            "Reservatório": res,
            "Cota Sangria": cota_sangria_val,
            col_anterior_label: nivel_anterior,  # valor de NÍVEL na data anterior (cabeçalho = dd/mm/aaaa)
            col_atual_label:    nivel_atual,     # valor de NÍVEL na data atual    (cabeçalho = dd/mm/aaaa)
            "Capacidade Total (m³)": cap_total,
            "Variação do Nível": variacao,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        order = ["Reservatório", "Cota Sangria", col_anterior_label, col_atual_label, "Capacidade Total (m³)", "Variação do Nível"]
        out = out.reindex(columns=order).sort_values("Reservatório").reset_index(drop=True)
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

try:
    if uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file, dtype=str)
    else:
        df_raw = load_data_from_url(url)

    # Filtro opcional por reservatório
    col_res_guess = find_column(df_raw, {"reservatorio", "reservatório", "acude", "açude", "nome"})
    if col_res_guess:
        reservatorios = sorted(x for x in df_raw[col_res_guess].dropna().unique())
        sel = st.multiselect("Filtrar reservatórios (opcional)", reservatorios, [])
        df_filtered = df_raw[df_raw[col_res_guess].isin(sel)] if sel else df_raw
    else:
        df_filtered = df_raw

    # Calcula a tabela final
    result = compute_table_global_dates(df_filtered)

    st.subheader("Resultado")
    if result.empty:
        st.info("Nenhum dado com as duas datas mais recentes foi encontrado.")
    else:
        # Formatação amigável pt-BR
        result_fmt = result.copy()
        for col in result_fmt.columns:
            if col == "Reservatório":
                continue
            if col == "Capacidade Total (m³)":
                result_fmt[col] = result_fmt[col].apply(
                    lambda v: f"{v:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".") if pd.notna(v) else ""
                )
            elif col in ("Cota Sangria", "Variação do Nível"):
                result_fmt[col] = result_fmt[col].apply(
                    lambda v: f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") if pd.notna(v) else ""
                )
            else:
                # Colunas com rótulos de data: exibem NÍVEL -> 2 casas
                result_fmt[col] = result_fmt[col].apply(
                    lambda v: f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") if pd.notna(v) else ""
                )

        st.dataframe(result_fmt, use_container_width=True, hide_index=True)

        # Download CSV bruto (sem formatação de string)
        csv_bytes = result.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Baixar CSV",
            data=csv_bytes,
            file_name="reservatorios_tabela_diaria.csv",
            mime="text/csv"
        )

        st.caption(
            "As colunas com datas no cabeçalho mostram o **Nível** nas **duas datas mais recentes** da planilha. "
            "• **Capacidade Total (m³)** = Volume do dia atual ÷ (Percentual do dia atual ÷ 100). "
            "• **Variação do Nível** = Nível (data atual) − Nível (data anterior)."
        )

except Exception as e:
    st.error(f"Ocorreu um erro ao processar os dados: {e}")
    st.stop()
