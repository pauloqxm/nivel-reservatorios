# app.py
import re
import math
import unicodedata
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
    """Converte URL do Google Sheets em URL CSV export."""
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url or "")
    gid_match = re.search(r"[#?]gid=(\d+)", url or "")
    gid = gid_match.group(1) if gid_match else None
    if not m:
        return url
    doc_id = m.group(1)
    base = f"https://docs.google.com/spreadsheets/d/{doc_id}/export?format=csv"
    return f"{base}&gid={gid}" if gid else base

def strip_accents_lower(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn').lower()

def to_number(x):
    """'1.234,56' -> 1234.56; preserva NaN."""
    if x is None:
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
    except Exception:
        return math.nan

def to_datetime_any(x):
    """Converte datas em vários formatos, retornando Timestamp normalizado (sem hora) ou NaT."""
    try:
        ts = pd.to_datetime(x, dayfirst=True, errors="coerce")
    except Exception:
        return pd.NaT
    return ts.normalize() if pd.notna(ts) else pd.NaT

@st.cache_data(ttl=900)
def load_data_from_url(url: str) -> pd.DataFrame:
    csv_url = google_sheets_to_csv_url(url)
    df = pd.read_csv(csv_url, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    return df

def find_column(df: pd.DataFrame, aliases):
    """Retorna o nome da coluna cujo normalizado bate com um dos aliases."""
    aliases = set(list(aliases))
    normalized = {col: strip_accents_lower(col) for col in df.columns}
    # Exato
    for col, norm in normalized.items():
        if norm in aliases:
            return col
    # Contém
    for col, norm in normalized.items():
        if any(alias in norm for alias in aliases):
            return col
    return None

# ==========================
# Núcleo de cálculo
# ==========================
def compute_table_global_dates(
    df_raw: pd.DataFrame,
    forced_prev_date: pd.Timestamp | None = None
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp, list[pd.Timestamp]]:
    """
    Retorna (df, data_anterior, data_atual, lista_datas_anteriores).
    Cabeçalhos de datas serão dd/mm/aaaa.
    """
    def last_scalar_on_date(dfr: pd.DataFrame, date_col: str, target_date, value_col: str) -> float:
        """Último valor numérico (float) para a data exata target_date. Sempre escalar ou NaN."""
        if pd.isna(target_date):
            return math.nan
        ddates = pd.to_datetime(dfr[date_col], errors="coerce").dt.normalize()
        sel = dfr.loc[ddates == pd.Timestamp(target_date).normalize(), value_col]
        if sel.empty:
            return math.nan
        sel = pd.to_numeric(sel, errors="coerce").dropna()
        if sel.empty:
            return math.nan
        return float(sel.iloc[-1])

    df = df_raw.copy()

    # Mapear colunas
    col_reservatorio = find_column(df, {"reservatorio", "reservatório", "acude", "açude", "nome"})
    col_cota_sangria = find_column(df, {"cota sangria", "cota de sangria", "cota_sangria", "cota excedencia"})
    col_data         = find_column(df, {"data", "dt", "dia"})
    col_volume       = find_column(df, {"volume", "vol"})
    col_percentual   = find_column(df, {"percentual", "perc", "percentual (%)", "volume (%)"})
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
            f"Faltando: {', '.join(missing)}. Ajuste os aliases no código ou renomeie na planilha."
        )

    # Conversões
    df[col_data]         = df[col_data].apply(to_datetime_any)
    df[col_volume]       = df[col_volume].apply(to_number)
    df[col_percentual]   = df[col_percentual].apply(to_number)
    df[col_nivel]        = df[col_nivel].apply(to_number)
    df[col_cota_sangria] = df[col_cota_sangria].apply(to_number)
    df = df.dropna(subset=[col_data])

    # Datas disponíveis (normalizadas, ordenadas)
    unique_dates = pd.to_datetime(df[col_data].dropna().unique(), errors="coerce")
    unique_dates = pd.Series(unique_dates).dropna().sort_values().unique().tolist()
    if not unique_dates:
        return pd.DataFrame(), pd.NaT, pd.NaT, []

    data_atual = unique_dates[-1]
    prev_candidates = [d for d in unique_dates if d < data_atual]
    prev_options_desc = prev_candidates[::-1]  # para calendário/menu

    if forced_prev_date and forced_prev_date in prev_candidates:
        data_anterior = forced_prev_date
    else:
        data_anterior = prev_candidates[-1] if prev_candidates else pd.NaT

    # Cabeçalhos com as datas (formatos dd/mm/aaaa)
    col_atual_label    = data_atual.strftime("%d/%m/%Y") if pd.notna(data_atual) else "Data Atual"
    col_anterior_label = data_anterior.strftime("%d/%m/%Y") if pd.notna(data_anterior) else "Data Anterior"

    rows = []
    for res, dfr in df.groupby(col_reservatorio, dropna=True):
        # Níveis
        nivel_atual    = last_scalar_on_date(dfr, col_data, data_atual,    col_nivel)
        nivel_anterior = last_scalar_on_date(dfr, col_data, data_anterior, col_nivel) if pd.notna(data_anterior) else math.nan

        # Volume/Percentual (dia atual) e Volume (dia anterior)
        vol_atual     = last_scalar_on_date(dfr, col_data, data_atual,    col_volume)
        vol_anterior  = last_scalar_on_date(dfr, col_data, data_anterior, col_volume) if pd.notna(data_anterior) else math.nan
        perc_atual    = last_scalar_on_date(dfr, col_data, data_atual, col_percentual)

        cap_total = vol_atual / (perc_atual / 100.0) if (pd.notna(vol_atual) and pd.notna(perc_atual) and perc_atual != 0) else math.nan
        variacao_nivel  = (nivel_atual - nivel_anterior) if (pd.notna(nivel_atual) and pd.notna(nivel_anterior)) else math.nan
        variacao_volume = (vol_atual  - vol_anterior)  if (pd.notna(vol_atual)  and pd.notna(vol_anterior))  else math.nan

        # Cota de sangria (preferir no dia atual; senão último histórico não-nulo)
        cota_atual = last_scalar_on_date(dfr, col_data, data_atual, col_cota_sangria)
        if pd.notna(cota_atual):
            cota_sangria_val = cota_atual
        else:
            cota_hist = pd.to_numeric(dfr[col_cota_sangria], errors="coerce").dropna()
            cota_sangria_val = float(cota_hist.iloc[-1]) if not cota_hist.empty else math.nan

        rows.append({
            "Reservatório": res,
            "Capacidade Total (m³)": cap_total,          # logo após Reservatório
            "Cota Sangria": cota_sangria_val,
            col_anterior_label: nivel_anterior,          # Nível (m) na data anterior
            col_atual_label:    nivel_atual,             # Nível (m) na data atual
            "Variação do Nível": variacao_nivel,
            "Variação do Volume": variacao_volume,       # NOVA COLUNA
            "Volume": vol_atual,
            "Percentual": perc_atual,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        order = [
            "Reservatório",
            "Capacidade Total (m³)",
            "Cota Sangria",
            col_anterior_label, col_atual_label,
            "Variação do Nível",
            "Variação do Volume",
            "Volume", "Percentual"
        ]
        out = out.reindex(columns=order).sort_values("Reservatório").reset_index(drop=True)

    return out, data_anterior, data_atual, prev_options_desc

# ==========================
# Renderização com grupos de cabeçalho e ícones
# ==========================
def format_ptbr(num, casas=2, inteiro=False):
    if pd.isna(num):
        return ""
    if inteiro:
        s = f"{num:,.0f}"
    else:
        s = f"{num:,.{casas}f}"
    return s.replace(",", "temp").replace(".", ",").replace("temp", ".")

def format_pct_br(num, casas=2):
    s = format_ptbr(num, casas=casas)
    return (s + "%") if s != "" else ""

def var_icon_html(v):
    """Número + seta: azul ▲ (positivo), vermelha ▼ (negativo), traço para zero/NaN."""
    if pd.isna(v):
        return ""
    val = format_ptbr(v, casas=2)
    if v > 0:
        return f'{val} <span style="color:#2563eb">▲</span>'
    if v < 0:
        return f'{val} <span style="color:#dc2626">▼</span>'
    return f"{val} —"

def render_table_with_group_headers(
    df: pd.DataFrame,
    prev_label: str,
    curr_label: str,
    volume_group_label: str,
    cota_group_label: str = "Cota (m)"
):
    """Renderiza uma tabela HTML com grupos mesclados de cabeçalho e ícones."""
    css = """
    <style>
    table.cota-table {width: 100%; border-collapse: collapse; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; font-size: 14px;}
    table.cota-table th, table.cota-table td {border: 1px solid #e5e7eb; padding: 8px 10px; text-align: right; white-space: nowrap;}
    table.cota-table th {background: #f8fafc; font-weight: 600; color: #111827;}
    table.cota-table td:first-child, table.cota-table th:first-child {text-align: left;}
    .group-head {text-align: center; background: #eef2ff;}
    </style>
    """
    html = [css, '<table class="cota-table">', "<thead>"]

    # Linha 1 do cabeçalho
    html.append("<tr>")
    html.append('<th rowspan="2">Reservatório</th>')
    html.append('<th rowspan="2">Capacidade Total (m³)</th>')
    html.append('<th rowspan="2">Cota Sangria</th>')
    html.append(f'<th class="group-head" colspan="2">{cota_group_label}</th>')
    html.append('<th rowspan="2">Variação do Nível</th>')
    html.append('<th rowspan="2">Variação do Volume</th>')
    html.append(f'<th class="group-head" colspan="2">{volume_group_label}</th>')
    html.append("</tr>")

    # Linha 2 do cabeçalho (subtítulos)
    html.append("<tr>")
    html.append(f"<th>{prev_label}</th>")
    html.append(f"<th>{curr_label}</th>")
    html.append("<th>Volume</th>")
    html.append("<th>Percentual (%)</th>")
    html.append("</tr>")
    html.append("</thead>")

    # Corpo
    html.append("<tbody>")
    for _, row in df.iterrows():
        html.append("<tr>")
        html.append(f"<td>{row['Reservatório']}</td>")
        html.append(f"<td>{format_ptbr(row['Capacidade Total (m³)'], casas=2)}</td>")  # 2 casas, ex.: 2,52
        html.append(f"<td>{format_ptbr(row['Cota Sangria'], casas=2)}</td>")
        html.append(f"<td>{format_ptbr(row[prev_label], casas=2)}</td>")
        html.append(f"<td>{format_ptbr(row[curr_label], casas=2)}</td>")
        html.append(f"<td>{var_icon_html(row['Variação do Nível'])}</td>")
        html.append(f"<td>{format_ptbr(row['Variação do Volume'], casas=2)}</td>")
        html.append(f"<td>{format_ptbr(row['Volume'], casas=2)}</td>")
        html.append(f"<td>{format_pct_br(row['Percentual'], casas=2)}</td>")
        html.append("</tr>")
    html.append("</tbody></table>")

    st.markdown("\n".join(html), unsafe_allow_html=True)

# ==========================
# UI (sem barra lateral) + Calendário no popover
# ==========================
st.title("📊 Tabela diária de Reservatórios")

try:
    # Carrega dados (sempre do Sheets)
    df_raw = load_data_from_url(SHEETS_URL)

    # Prévia opcional
    with st.expander("Visualizar dados brutos"):
        st.dataframe(df_raw.head(), use_container_width=True)

    # Lê data anterior selecionada da URL (?prev=YYYY-MM-DD)
    q = st.query_params
    forced_prev = None
    if "prev" in q and q["prev"]:
        try:
            forced_prev = pd.to_datetime(q["prev"], errors="coerce").normalize()
        except Exception:
            forced_prev = None

    # Filtro opcional por reservatório
    col_res_guess = find_column(df_raw, {"reservatorio", "reservatório", "acude", "açude", "nome"})
    if col_res_guess:
        reservatorios = sorted(x for x in df_raw[col_res_guess].dropna().unique() if x)
        sel = st.multiselect("Filtrar reservatórios (opcional)", reservatorios, [], placeholder="Selecione…")
        df_filtered = df_raw[df_raw[col_res_guess].isin(sel)] if sel else df_raw
    else:
        df_filtered = df_raw
        st.warning("Não foi possível identificar a coluna de Reservatório.")

    # Processa
    with st.spinner("Processando dados..."):
        result, dprev, dcurr, prev_options_desc = compute_table_global_dates(df_filtered, forced_prev_date=forced_prev)

    st.subheader("Resultado")

    # ===== Toolbar com calendário (popover) =====
    if prev_options_desc:
        min_prev = prev_options_desc[-1].date()
        max_prev = prev_options_desc[0].date()
    else:
        min_prev = None
        max_prev = None

    left, right = st.columns([1, 3])
    with left:
        label_btn = f"📅 Alterar data anterior: {dprev.strftime('%d/%m/%Y') if pd.notna(dprev) else '—'}"
        with st.popover(label_btn, use_container_width=True):
            st.markdown("**Escolha a data anterior**")
            default_date = dprev.date() if pd.notna(dprev) else (max_prev or pd.Timestamp.today().date())
            date_sel = st.date_input(
                "Data anterior",
                value=default_date,
                min_value=min_prev,
                max_value=(dcurr - pd.Timedelta(days=1)).date() if pd.notna(dcurr) else None,
                format="DD/MM/YYYY"
            )
            if st.button("Aplicar", type="primary"):
                st.query_params.update({"prev": pd.Timestamp(date_sel).strftime("%Y-%m-%d")})
                st.rerun()

    # ===== Tabela =====
    if result.empty:
        st.info("Nenhum dado encontrado para as datas selecionadas.")
    else:
        prev_label = dprev.strftime("%d/%m/%Y") if pd.notna(dprev) else "Data Anterior"
        curr_label = dcurr.strftime("%d/%m/%Y") if pd.notna(dcurr) else "Data Atual"
        volume_group_label = f"Volume ({curr_label})"

        render_table_with_group_headers(
            result,
            prev_label=prev_label,
            curr_label=curr_label,
            volume_group_label=volume_group_label,
            cota_group_label="Cota (m)"
        )

        # ===== CSV (formatado) =====
        csv_df = result.copy()
        desired_cols = [
            "Reservatório",
            "Capacidade Total (m³)",
            "Cota Sangria",
            prev_label, curr_label,
            "Variação do Nível",
            "Variação do Volume",
            "Volume", "Percentual"
        ]

        def fmt2(v):  # 2 casas com vírgula
            return format_ptbr(v, casas=2)

        csv_df["Capacidade Total (m³)"] = csv_df["Capacidade Total (m³)"].apply(fmt2)  # duas casas
        csv_df["Cota Sangria"] = csv_df["Cota Sangria"].apply(fmt2)
        csv_df[prev_label] = csv_df[prev_label].apply(fmt2)
        csv_df[curr_label] = csv_df[curr_label].apply(fmt2)
        csv_df["Variação do Nível"] = csv_df["Variação do Nível"].apply(fmt2)
        csv_df["Variação do Volume"] = csv_df["Variação do Volume"].apply(fmt2)
        csv_df["Volume"] = csv_df["Volume"].apply(fmt2)
        csv_df["Percentual"] = csv_df["Percentual"].apply(lambda v: format_pct_br(v, casas=2))

        csv_df = csv_df[[c for c in desired_cols if c in csv_df.columns]]
        csv_bytes = csv_df.to_csv(index=False, sep=';', decimal=',').encode("utf-8")
        st.download_button("⬇️ Baixar CSV (formatado)", data=csv_bytes,
                           file_name="reservatorios_tabela_diaria.csv",
                           mime="text/csv")

        st.caption(
            "• **Capacidade Total (m³)** com **duas casas decimais** (ex.: 2,52). "
            "• **Variação do Nível**: seta **azul (▲)** para positivo, **vermelha (▼)** para negativo. "
            "• **Variação do Volume**: diferença entre Volume (data atual) e Volume (data anterior). "
            f"• Cabeçalhos agrupados: **Cota (m)** → {prev_label} e {curr_label}; **{volume_group_label}** → Volume e Percentual."
        )

except Exception as e:
    st.error(f"Ocorreu um erro ao processar os dados: {str(e)}")
    import traceback
    with st.expander("Detalhes do erro"):
        st.code(traceback.format_exc())
