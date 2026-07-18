"""
Dashboard monitoringu jakości potoku transkrypcji ASR.
Źródła: metrics_results/metrics_log.csv (jakość, per przebieg)
        metrics_results/performance.csv (wydajność, per eksperyment)
Uruchomienie: streamlit run dashboard.py
"""
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="ASR Pipeline Monitor", layout="wide")
st.title("Monitoring jakości potoku transkrypcji ASR")
st.caption("Projekt magisterski — porównanie potoków Whisper / Faster-Whisper "
           "| źródła: metrics_log.csv (jakość) + performance.csv (wydajność)")

# --- Wczytanie i złączenie źródeł (blend po kluczu 'eksperyment') ---
quality = pd.read_csv("metrics_results/metrics_log.csv", sep=";")
quality = quality[quality["eksperyment"] != "faster_whisper_vad_slm"] 
perf = pd.read_csv("metrics_results/performance.csv", sep=";")
quality["data"] = pd.to_datetime(quality["data"])
latest = (quality.sort_values("data").groupby("eksperyment").tail(1))
df = latest.merge(perf, on="eksperyment", how="left")

# --- Filtry (sidebar) ---
st.sidebar.header("Filtry")
exps = st.sidebar.multiselect("Eksperymenty", sorted(df["eksperyment"]),
                              default=sorted(df["eksperyment"]))
df = df[df["eksperyment"].isin(exps)]
q_hist = quality[quality["eksperyment"].isin(exps)]

# --- KPI (scorecardy) ---
best = df.loc[df["wer_norm"].idxmin()]
k1, k2, k3, k4 = st.columns(4)
k1.metric("Najlepszy WER", f"{best['wer_norm']:.2f}%", best["eksperyment"])
k2.metric("Najlepszy RTF", f"{df['rtf'].min():.3f}",
          df.loc[df["rtf"].idxmin(), "eksperyment"])
k3.metric("Najniższy RAM (peak)", f"{df['ram_peak_mb'].min():.0f} MB",
          df.loc[df["ram_peak_mb"].idxmin(), "eksperyment"])
k4.metric("Liczba przebiegów w logu", len(q_hist))

# --- Sekcja 1: Jakość ---
st.header("1. Jakość transkrypcji")
c1, c2 = st.columns(2)
c1.plotly_chart(px.bar(df, x="eksperyment", y=["wer_norm", "wer_raw", "cer"],
                barmode="group", title="WER / WER raw / CER per eksperyment (%)"),
                use_container_width=True)
err = df.melt(id_vars="eksperyment", value_vars=["S", "D", "I"],
              var_name="typ błędu", value_name="liczba")
c2.plotly_chart(px.bar(err, x="eksperyment", y="liczba", color="typ błędu",
                title="Rozkład błędów: substytucje / delecje / insercje"),
                use_container_width=True)

# --- Sekcja 2: Wydajność ---
st.header("2. Wydajność i zasoby")
c3, c4 = st.columns(2)
c3.plotly_chart(px.bar(df, x="eksperyment", y="rtf",
                title="RTF (poniżej 1.0 = szybciej niż czas rzeczywisty)")
                .add_hline(y=1.0, line_dash="dash", line_color="red"),
                use_container_width=True)
c4.plotly_chart(px.scatter(df, x="rtf", y="wer_norm", size="ram_peak_mb",
                text="eksperyment", title="Jakość vs wydajność (rozmiar = RAM peak)")
                .update_traces(textposition="top center"),
                use_container_width=True)

# --- Sekcja 3: Historia przebiegów ---
st.header("3. Historia przebiegów (drill-down)")
st.plotly_chart(px.line(q_hist.sort_values("data"), x="data", y="wer_norm",
                color="eksperyment", markers=True,
                title="WER w kolejnych przebiegach ewaluacji"),
                use_container_width=True)
st.dataframe(q_hist.sort_values("data", ascending=False), use_container_width=True)

st.caption("README: WER/CER liczone jiwer (normalizacja: lowercase, bez interpunkcji) "
           "względem ręcznego ground truth (422 słowa). RTF = czas przetwarzania / "
           "długość nagrania. Blend źródeł po kluczu 'eksperyment'.")