import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats

st.set_page_config(
    page_title="EDA cu Streamlit Baicu Cosmin-Mihai",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 TEMA: EDA cu Streamlit Baicu Cosmin-Mihai")

# ---------- Helpers ----------
@st.cache_data(show_spinner=False)
def read_file(uploaded_file: "st.runtime.uploaded_file_manager.UploadedFile") -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        try:
            return pd.read_csv(uploaded_file)
        except Exception:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, sep=";")
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    raise ValueError("Format invalid. Încarcă CSV sau Excel (.xlsx/.xls).")

def detect_numeric_cols(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()

def detect_categorical_cols(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

def iqr_outlier_stats(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    rows = []
    n = len(df)
    for col in numeric_cols:
        s = df[col].dropna()
        if len(s) == 0:
            rows.append({"Coloană": col, "Outlieri": 0, "Procent": 0.0, "Q1": np.nan, "Q3": np.nan,
                         "IQR": np.nan, "Lower": np.nan, "Upper": np.nan})
            continue
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        pct = (outliers / n * 100) if n else 0.0
        rows.append({"Coloană": col, "Outlieri": int(outliers), "Procent": float(pct),
                     "Q1": float(q1), "Q3": float(q3), "IQR": float(iqr), "Lower": float(lower), "Upper": float(upper)})
    return pd.DataFrame(rows).sort_values(["Outlieri", "Procent"], ascending=False)

# ---------- Upload ----------
st.sidebar.header("1) Încărcare date")
uploaded = st.sidebar.file_uploader("Încarcă un fișier CSV sau Excel", type=["csv", "xlsx", "xls"])

if uploaded is None:
    st.info("Încarcă un fișier CSV sau Excel din sidebar pentru a începe.")
    st.stop()

try:
    df = read_file(uploaded)
    st.sidebar.success("✅ Fișier citit corect!")
except Exception as e:
    st.sidebar.error(f"❌ Nu am putut citi fișierul: {e}")
    st.stop()

# Basic validation
if df is None or df.empty:
    st.error("Fișierul a fost citit, dar datasetul este gol.")
    st.stop()

st.subheader("✅ Primele 10 rânduri")
st.dataframe(df.head(10), use_container_width=True)

numeric_cols = detect_numeric_cols(df)
cat_cols = detect_categorical_cols(df)

# ---------- Tabs ----------
tab_filter, tab_overview, tab_numeric, tab_cat, tab_corr = st.tabs(
    ["Cerința 1: Filtrare", "Cerința 2: Overview", "Cerința 3: Numeric", "Cerința 4: Categoric", "Cerința 5: Corelații & Outlieri"]
)

# ---------- Cerința 1 ----------
with tab_filter:
    st.header("Cerința 1 — Încărcare + Filtrare")
    st.write("Dataset încărcat. Configurează filtrele de mai jos:")

    df_before = df.copy()
    df_f = df.copy()

    st.markdown("### Filtrare coloane numerice (slidere)")
    if numeric_cols:
        with st.expander("Filtre numerice", expanded=True):
            for col in numeric_cols:
                col_min = float(np.nanmin(df[col].values))
                col_max = float(np.nanmax(df[col].values))
                if np.isfinite(col_min) and np.isfinite(col_max) and col_min != col_max:
                    rng = st.slider(
                        f"{col}",
                        min_value=col_min,
                        max_value=col_max,
                        value=(col_min, col_max),
                        step=(col_max - col_min) / 200 if (col_max - col_min) > 0 else 1.0,
                        key=f"num_{col}",
                    )
                    df_f = df_f[df_f[col].between(rng[0], rng[1]) | df_f[col].isna()]
                else:
                    st.caption(f"⚠️ {col}: nu poate fi filtrată (valori constante sau lipsesc).")
    else:
        st.info("Nu există coloane numerice în dataset.")

    st.markdown("### Filtrare coloane categorice (multiselect)")
    if cat_cols:
        with st.expander("Filtre categorice", expanded=True):
            for col in cat_cols:
                uniques = df[col].dropna().astype(str).unique().tolist()
                uniques_sorted = sorted(uniques)[:200]
                selected = st.multiselect(
                    f"{col} (alege valori; gol = fără filtrare)",
                    options=uniques_sorted,
                    default=[],
                    key=f"cat_{col}",
                )
                if selected:
                    df_f = df_f[df_f[col].astype(str).isin(selected)]
    else:
        st.info("Nu există coloane categorice în dataset.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Rânduri înainte", len(df_before))
    with c2:
        st.metric("Rânduri după", len(df_f))
    with c3:
        delta = len(df_f) - len(df_before)
        st.metric("Δ rânduri", delta)

    st.markdown("### DataFrame filtrat")
    st.dataframe(df_f, use_container_width=True)

# ---------- Cerința 2 ----------
with tab_overview:
    st.header("Cerința 2 — Overview, tipuri de date, lipsă, statistici")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Număr rânduri", len(df))
    with c2:
        st.metric("Număr coloane", df.shape[1])

    st.subheader("Tipuri de date pe coloană")
    dtype_df = pd.DataFrame({
        "Coloană": df.columns,
        "Tip": df.dtypes.astype(str),
        "Non-Null": df.count().values,
        "Null": df.isnull().sum().values
    })
    st.dataframe(dtype_df, use_container_width=True)

    st.subheader("Valori lipsă pe coloană")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    miss_df = pd.DataFrame({"Coloană": missing.index, "Număr lipsă": missing.values, "Procent (%)": missing_pct.values})
    miss_df = miss_df[miss_df["Număr lipsă"] > 0].sort_values("Număr lipsă", ascending=False)

    if miss_df.empty:
        st.success("✅ Nu există valori lipsă în dataset.")
    else:
        st.dataframe(miss_df, use_container_width=True)
        fig = px.bar(miss_df, x="Coloană", y="Procent (%)", text="Număr lipsă", title="Procent valori lipsă per coloană")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Statistici descriptive (coloane numerice)")
    if numeric_cols:
        desc = df[numeric_cols].describe().T
        desc["median"] = df[numeric_cols].median(numeric_only=True)
        cols_order = ["count", "mean", "median", "std", "min", "25%", "50%", "75%", "max"]
        cols_order = [c for c in cols_order if c in desc.columns]
        st.dataframe(desc[cols_order], use_container_width=True)
    else:
        st.info("Nu există coloane numerice pentru statistici descriptive.")

# ---------- Cerința 3 ----------
with tab_numeric:
    st.header("Cerința 3 — Histogramă + bins slider + boxplot + statistici")
    if not numeric_cols:
        st.info("Nu există coloane numerice în dataset.")
    else:
        col = st.selectbox("Selectează o coloană numerică", numeric_cols, key="c3_numeric_col")
        bins = st.slider("Număr de bins", 10, 100, 30, 1, key="c3_bins")

        s = df[col].dropna()
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Medie", f"{s.mean():.4g}" if len(s) else "NA")
        with c2:
            st.metric("Mediană", f"{s.median():.4g}" if len(s) else "NA")
        with c3:
            st.metric("Deviație std", f"{s.std():.4g}" if len(s) else "NA")

        fig_h = px.histogram(df, x=col, nbins=bins, title=f"Histogramă: {col} (bins={bins})")
        st.plotly_chart(fig_h, use_container_width=True)

        fig_b = px.box(df, y=col, points="outliers", title=f"Box plot: {col}")
        st.plotly_chart(fig_b, use_container_width=True)

# ---------- Cerința 4 ----------
with tab_cat:
    st.header("Cerința 4 — Coloane categorice + count plot + frecvențe")
    if not cat_cols:
        st.info("Nu există coloane categorice în dataset.")
    else:
        col = st.selectbox("Selectează o coloană categorică", cat_cols, key="c4_cat_col")
        vc = df[col].astype(str).value_counts(dropna=False)
        freq_df = pd.DataFrame({
            "Valoare": vc.index.astype(str),
            "Frecvență": vc.values,
            "Procent (%)": (vc.values / len(df) * 100).round(2)
        })

        top_n = st.slider("Top N valori (pentru grafic)", 5, 50, 20, 1, key="c4_topn")
        fig = px.bar(
            freq_df.head(top_n),
            x="Valoare",
            y="Frecvență",
            text="Procent (%)",
            title=f"Count plot (Top {top_n}): {col}"
        )
        fig.update_traces(textposition="outside")
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Tabel frecvențe absolute și procente")
        st.dataframe(freq_df, use_container_width=True)

# ---------- Cerința 5 ----------
with tab_corr:
    st.header("Cerința 5 — Corelații, scatter + Pearson, outlieri IQR")
    if not numeric_cols:
        st.info("Nu există coloane numerice pentru corelații / outlieri.")
    else:
        st.subheader("Matrice de corelație + heatmap")
        corr = df[numeric_cols].corr(numeric_only=True)
        fig_hm = px.imshow(corr, text_auto=True, aspect="auto", title="Heatmap corelații (Pearson)")
        st.plotly_chart(fig_hm, use_container_width=True)

        st.subheader("Scatter plot + coeficient Pearson")
        c1, c2 = st.columns(2)
        with c1:
            x = st.selectbox("Variabila X", numeric_cols, key="c5_x")
        with c2:
            y = st.selectbox("Variabila Y", numeric_cols, index=min(1, len(numeric_cols)-1), key="c5_y")

        scatter_df = df[[x, y]].dropna()
        fig_sc = px.scatter(scatter_df, x=x, y=y, trendline=None, title=f"Scatter: {x} vs {y}")
        st.plotly_chart(fig_sc, use_container_width=True)

        if len(scatter_df) >= 3:
            r, p = stats.pearsonr(scatter_df[x], scatter_df[y])
            st.info(f"Coeficient Pearson r = **{r:.4f}**, p-value = **{p:.4g}** (n={len(scatter_df)})")
        else:
            st.warning("Prea puține date non-NaN pentru Pearson (minim 3 rânduri).")

        st.subheader("Outlieri (IQR) — număr și procent pentru fiecare coloană numerică")
        out_stats = iqr_outlier_stats(df, numeric_cols)
        st.dataframe(out_stats[["Coloană", "Outlieri", "Procent", "Lower", "Upper"]], use_container_width=True)

        st.subheader("Vizualizare outlieri pe grafic")
        col = st.selectbox("Alege coloană numerică pentru outlieri", numeric_cols, key="c5_out_col")

        # compute fences for selected
        s = df[col].dropna()
        if len(s) == 0:
            st.warning("Coloana selectată nu are valori numerice disponibile.")
        else:
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            fig_box = px.box(df, y=col, points="outliers", title=f"Box plot cu outlieri (IQR): {col}")
            fig_box.add_hline(y=lower, line_dash="dash", annotation_text="Lower (Q1-1.5*IQR)")
            fig_box.add_hline(y=upper, line_dash="dash", annotation_text="Upper (Q3+1.5*IQR)")
            st.plotly_chart(fig_box, use_container_width=True)

            out_count = ((df[col] < lower) | (df[col] > upper)).sum()
            out_pct = out_count / len(df) * 100
            st.write(f"Outlieri în **{col}**: **{int(out_count)}** ({out_pct:.2f}%)")
