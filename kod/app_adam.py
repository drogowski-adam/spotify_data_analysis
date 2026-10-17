import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

#st.set_page_config(page_title="Dzień vs noc", layout="wide")

FILES_DAY_NIGHT = {
    "Basia": "dane/basia_moods.csv",
    "Paweł": "dane/pawel_moods.csv",
    "Adam": "dane/adam_moods.csv",
}


def load_df_adam(path: str) -> pd.DataFrame:

    df = pd.read_csv(path)

    required = {"ts", "ms_played", "valence", "danceability"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")

    df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    df = df.dropna(subset=["ts"])

    df["hour"] = df["ts"].dt.hour
    df["month"] = df["ts"].dt.to_period("M").dt.to_timestamp()
    df["minutes"] = df["ms_played"] / 60000.0
    df["melancholy"] = 1.0 - df["valence"]

    return df


def hour_agg_adam(df: pd.DataFrame) -> pd.DataFrame:

    agg = (
        df.groupby("hour", as_index=False)
        .agg(
            minutes_sum=("minutes", "sum"),
            plays=("ms_played", "size"),
            mood_mean=("valence", "mean"),
            dance_mean=("danceability", "mean"),
            mel_mean=("melancholy", "mean"),
        )
    )

    total_minutes = agg["minutes_sum"].sum()
    agg["minutes_pct"] = (agg["minutes_sum"] / total_minutes) * 100

    base = pd.DataFrame({"hour": list(range(24))})
    return base.merge(agg, on="hour", how="left")


def tile_row_adam(ax, values, title):

    mat = np.array(values, dtype=float).reshape(1, 24)
    im = ax.imshow(mat, aspect="equal", cmap="Blues")

    ax.set_yticks([])
    ax.set_title(title, fontsize=12, pad=10)

    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}" for h in range(24)], fontsize=9)

    ax.set_xticks(np.arange(-.5, 24, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 1, 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=0.6, color="#444")

    return im


def line_row_adam(ax, hours, values, title, color):

    ax.plot(hours, values, marker="o", linewidth=2.5, color=color)
    ax.set_xlim(0, 23)
    ax.set_xticks(range(24))
    ax.set_title(title, fontsize=12, pad=10)
    ax.grid(True, alpha=0.25)
    ax.set_facecolor("none")


def render_dashboard_adam(agg: pd.DataFrame, person: str, start_month, end_month):

    agg = agg.set_index("hour").reindex(range(24)).reset_index()

    plt.style.use("dark_background")
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(18, 9), constrained_layout=True)

    fig.suptitle(
        f"{person} | {start_month:%Y-%m} → {end_month:%Y-%m}",
        fontsize=15
    )

    im1 = tile_row_adam(
        axes[0],
        agg["minutes_pct"].values,
        "% czasu spędzonego na Spotify w tej godzinie"
    )

    c1 = fig.colorbar(im1, ax=axes[0], fraction=0.02, pad=0.02)
    c1.set_label("% czasu")

    hours = agg["hour"].values

    line_row_adam(axes[1], hours, agg["mood_mean"], "Mood (valence)", "#00D1FF")
    line_row_adam(axes[2], hours, agg["dance_mean"], "Danceability", "#00FF88")
    line_row_adam(axes[3], hours, agg["mel_mean"], "Melancholijność", "#FF4D6D")

    axes[3].set_xticklabels([f"{h:02d}" for h in range(24)])

    st.pyplot(fig)


#st.title("Dzień vs noc — profil 24h")

#person = st.sidebar.selectbox("Osoba", list(FILES.keys()))

#df = load_df_adam(FILES[person])

#months = sorted(df["month"].dropna().unique())
#if not months:
    #st.error("Brak danych miesięcznych.")
    #st.stop()

#c1, c2 = st.columns(2)
#with c1:
    #start_month = st.selectbox("Od miesiąca", months, index=0)
#with c2:
    #end_month = st.selectbox("Do miesiąca", months, index=len(months) - 1)

#if start_month > end_month:
    #start_month, end_month = end_month, start_month

#df_sel = df[(df["month"] >= start_month) & (df["month"] <= end_month)].copy()

#total_hours = df_sel["minutes"].sum() / 60

#st.markdown(
    #f"""
#**Zakres:** {start_month:%Y-%m} → {end_month:%Y-%m}  
#**Streamy:** {len(df_sel):,}  
#**Łączny czas słuchania:** {total_hours:.1f} godzin
#"""
#)

#agg = hour_agg_adam(df_sel)

#render_dashboard_adam(agg, person, start_month, end_month)

#st.subheader("Agregacja per godzina")
#st.dataframe(
    #agg.set_index("hour").reindex(range(24)).reset_index().style.format({
        #"minutes_pct": "{:.2f}",
        #"plays": "{:.0f}",
        #"mood_mean": "{:.3f}",
        #"dance_mean": "{:.3f}",
        #"mel_mean": "{:.3f}",
    #}),
    #use_container_width=True
#)