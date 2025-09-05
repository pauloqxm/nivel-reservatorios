import io
import re
import math
import unicodedata
from datetime import datetime
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Reservatórios – Tabela diária", layout="wide")

SHEETS_URL = "https://docs.google.com/spreadsheets/d/1zZ0RCyYj-AzA_dhWzxRziDWjgforbaH7WIoSEd2EKdk/edit?gid=1305065127#gid=1305065127"

@st.cache_data(ttl=900)
def google_sheets_to_csv_url(url: str) -> str:
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
    gid = None
    gid_match = re.search(r"[#?]gid=(\d+)", url)
    if gid_match:
        gid = gid_match.group(1)
    if not m:
        return url
    doc_id = m.group(1)
    if gid is None:
        return f"https://docs.google.com/spreadsheets/d/{doc_id}/export?format=csv"
    return f"https://docs.google.com/spreadsheets/d/{doc_id}/export?format=csv&gid={gid}"

def strip_accents_lower(s: str) -> str:
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    return s.lower()

def to_number(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return math.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return math.nan
    s = ''.join(ch for ch in s if ch.isdigit() or ch in ".,-")
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return math.nan

def to_datetime_any(x):
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
    normalized = {col: strip_accents_lower(col) for col in df.columns}
    if isinstance(aliases, set):
        aliases = list(aliases)
    aliases = set(aliases)
    for col, norm in normalized.items():
        if norm in aliases:
            return col
    for col, norm in normalized.items():
        if any(alias in norm for alias in aliases):
            return col
    return None

def compute_table_global_dates(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # Mapear colunas
    col_reservatorio = find_column(df, {"reservatorio", "reservatório", "acude", "açude", "nome"})
    col_cota_sangria = find_column(df, {"cota sangria", "cota de sangria", "cota_sangria", "cota excedencia"})
    col_data = find_column(df, {"data", "dt", "dia"})
    col_volume = find_column(df, {"volume", "vol"})
    col_percentual = find_column(df, {"percentual", "perc", "percentual (%)", "volume (%)"})
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

    # Conversões
    df[col_data] = df[col_data].apply(to_datetime_any)
    df[col_volume] = df[col_volume].apply(to_number)
    df[col_percentual] = df[col_percentual].apply(to_number)
    df[col_nivel] = df[col_nivel].apply(to_number)
    df[col_cota_sangria] = df[col_cota_sangria].apply(to_number)
    df = df.dropna(subset=[col_data])

    # Determinar as DUAS datas mais recentes globais
    unique_dates = sorted(df[col_data].dropna().unique())
    if len(unique_dates) == 0:
        return pd.DataFrame()
    data_atual = pd.to_datetime(unique_dates[-1])
    data_anterior = pd.to_datetime(unique_dates[-2]) if len(unique_dates) >= 2 else pd.NaT

    # Cabeçalhos de coluna com as datas no formato dd/mm/aaaa
    col_atual_label = data_atual.strftime("%d/%m/%Y") if pd.notna(data_atual) else "Data Atual"
    col_anterior_label = data_anterior.strftime("%d/%m/%Y") if pd.notna(data_anterior) else "Data Anterior"

    # Preparar saída
    rows = []
    for res, dfr in df.groupby(col_reservatorio, dropna=True):
        # pegar nível nas datas globais
        nivel_atual = dfr.loc[dfr[col_data] == data_atual, col_nivel].dropna()
        nivel_atual = float(nivel_atual.iloc[-1]) if not nivel_atual.empty else math.nan

        if pd.notna(data_anterior):
            nivel_anterior = dfr.loc[dfr[col_data] == data_anterior, col_nivel].dropna()
            nivel_anterior = float(nivel_anterior.iloc[-1]) if not nivel_anterior.empty else math.nan
        else:
            nivel_anterior = math.nan

        # capacidade total = Volume / (Percentual/100) usando a linha do dia ATUAL (global)
        vol_atual = dfr.loc[dfr[col_data] == data_atual, col_volume].dropna()
        vol_atual = float(vol_atual.iloc[-1]) if not vol_atual.empty else math.nan

        perc_atual = dfr.loc[dfr[col_data] == data_atual, col_percentual].dropna()
        perc_atual = float(perc_atual.iloc[-1]) if not perc_atual.empty else math.nan

        if perc_atual and not math.isnan(perc_atual) and perc_atual != 0:
            cap_total = vol_atual / (perc_atual / 100.0) if vol_atual and not math.isnan(vol_atual) else math.nan
        else:
            cap_total = math.nan

        # cota de sangria (da linha do dia atual, se existir; senão último valor conhecido)
        cota_s_atual = dfr.loc[dfr[col_data] == data_atual, col_cota_sangria].dropna()
        if not cota_s_atual.empty:
            cota_sangria_val = float(cota_s_atual.iloc[-1])
        else:
            # fallback: último valor não-nulo
            cota_s_hist = dfr[col_cota_sangria].dropna()
            cota_sangria_val = float(cota_s_hist.iloc[-1]) if not cota_s_hist.empty else math.nan

        # variação (nível atual - nível anterior)
        variacao = (nivel_atual - nivel_anterior) if (not math.isnan(nivel_atual) and not math.isnan(nivel_anterior)) else math.nan

        row = {
            "Reservatório": res,
            "Cota Sangria": cota_sangria_val,
            col_anterior_label: nivel_anterior,  # << Nível na data anterior
            col_atual_label: nivel_atual,        # << Nível na data atual
            "Capacidade Total (m³)": cap_total,
            "Variação do Nível": variacao,
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        # Ordena pelas colunas principais
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

    # filtro opcional antes do cálculo
    col_res_guess = find_column(df_raw, {"reservatorio", "reservatório", "acude", "açude", "nome"})
    if col_res_guess:
        reservatorios = sorted(x for x in df_raw[col_res_guess].dropna().unique())
        sel = st.multiselect("Filtrar reservatórios (opcional)", reservatorios, [])
        df_filtered = df_raw[df_raw[col_res_guess].isin(sel)] if sel else df_raw
    else:
        df_filtered = df_raw

    result = compute_table_global_dates(df_filtered)

    st.subheader("Resultado")
    if result.empty:
        st.info("Nenhum dado com as duas datas mais recentes foi encontrado.")
    else:
        # Formatação
        result_fmt = result.copy()
        # números com separador pt-BR
        for col in result_fmt.columns:
            if col in ("Reservatório",):
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
                # As colunas de data (cabeçalho com dd/mm/aaaa) contêm Nível → 2 casas
                result_fmt[col] = result_fmt[col].apply(
                    lambda v: f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") if pd.notna(v) else ""
                )

        st.dataframe(result_fmt, use_container_width=True, hide_index=True)

        # CSV sem formatação
        csv_bytes = result.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Baixar CSV",
            data=csv_bytes,
            file_name="reservatorios_tabela_diaria.csv",
            mime="text/csv"
        )

        st.caption(
            "As colunas com datas no cabeçalho mostram o **Nível** de cada reservatório na **data anterior** e na **data atual** "
            "(duas datas mais recentes da planilha). • **Capacidade Total (m³)** = Volume do dia atual ÷ (Percentual do dia atual ÷ 100). "
            "• **Variação do Nível** = Nível (data atual) − Nível (data anterior)."
        )

except Exception as e:
    st.error(f"Ocorreu um erro ao processar os dados: {e}")
    st.stop()
