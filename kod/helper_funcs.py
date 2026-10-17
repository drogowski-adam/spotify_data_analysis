import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from theme import THEME


@st.cache_data
def load_main_data():

    files = {
        "Adam": "dane/dane_adam.csv",
        "Basia": "dane/dane_basia.csv",
        "Paweł": "dane/dane_pawel.csv"
    }

    dfs = []
    for user, path in files.items():
        try:
            df = pd.read_csv(path)
            df['user'] = user
            dfs.append(df)
        except FileNotFoundError:
            st.error(f"Brakuje pliku głównego: {path}")
            return pd.DataFrame()

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    df['ts'] = pd.to_datetime(df['ts'], errors='coerce')
    df = df.dropna(subset=['ts'])
    df['hour'] = df['ts'].dt.hour
    
    df['master_metadata_album_artist_name'] = df['master_metadata_album_artist_name'].fillna('Nieznany Artysta')
    df['master_metadata_track_name'] = df['master_metadata_track_name'].fillna('Nieznany Utwór')
    df['platform'] = df['platform'].fillna('Inne')
    
    return df


@st.cache_data
def load_quotes():
    try:
        return pd.read_csv("dane/awful_quotes.csv")
    except FileNotFoundError:
        return pd.DataFrame(columns=['user', 'artist', 'track', 'quote'])


MOOD_FILES = {
    "Adam": "dane/adam_moods.csv",
    "Basia": "dane/basia_moods.csv",
    "Paweł": "dane/pawel_moods.csv",
}


def load_mood_df(path: str) -> pd.DataFrame:

    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame()

    df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    df = df.dropna(subset=["ts"])
    df["hour"] = df["ts"].dt.hour
    df["month"] = df["ts"].dt.to_period("M").dt.to_timestamp()
    df["minutes"] = df["ms_played"] / 60000.0
    df["melancholy"] = 1.0 - df["valence"]

    return df


def hour_agg(df: pd.DataFrame) -> pd.DataFrame:

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

    base = pd.DataFrame({"hour": list(range(24))})
    return base.merge(agg, on="hour", how="left")


def tight_scale(values: pd.Series, pad: float = 0.02):

    v = values.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(v) == 0:
        return 0.0, 1.0
    lo = float(v.min())
    hi = float(v.max())
    if lo == hi:
        lo -= 0.01
        hi += 0.01
    lo = max(0.0, lo - pad)
    hi = min(1.0, hi + pad)

    return lo, hi


def tile_row(ax, values, title, cmap, vmin=None, vmax=None, show_xticks=False):

    mat = np.array(values, dtype=float).reshape(1, 24)
    im = ax.imshow(mat, aspect="equal", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_yticks([])
    ax.set_title(title, fontsize=12, pad=10)

    if show_xticks:
        ax.set_xticks(range(24))
        ax.set_xticklabels([f"{h:02d}" for h in range(24)], fontsize=9)
    else:
        ax.set_xticks([])

    ax.set_xticks(np.arange(-.5, 24, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 1, 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=0.9)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im


def render_dashboard(agg: pd.DataFrame, person: str, start_month, end_month):

    agg = agg.set_index("hour").reindex(range(24)).reset_index()
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(18, 8), constrained_layout=True)
    
    fig.suptitle(
        f"Dzień vs noc — 24h | {person} | {start_month:%Y-%m} → {end_month:%Y-%m}",
        fontsize=14
    )

    im1 = tile_row(axes[0], agg["minutes_sum"].values, "Minuty słuchania (suma)", "viridis")
    
    mood_vmin, mood_vmax = tight_scale(agg["mood_mean"])
    im2 = tile_row(axes[1], agg["mood_mean"].values, "Mood (valence)", "viridis", vmin=mood_vmin, vmax=mood_vmax)

    dance_vmin, dance_vmax = tight_scale(agg["dance_mean"])
    im3 = tile_row(axes[2], agg["dance_mean"].values, "Danceability", "viridis", vmin=dance_vmin, vmax=dance_vmax)

    mel_vmin, mel_vmax = tight_scale(agg["mel_mean"])
    im4 = tile_row(axes[3], agg["mel_mean"].values, "Melancholia (1 - valence)", "viridis", vmin=mel_vmin, vmax=mel_vmax, show_xticks=True)

    for ax, im, label in zip(axes, [im1, im2, im3, im4], ["Minuty", "Mood", "Dance", "Melancholia"]):
        cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
        cbar.set_label(label)

    st.pyplot(fig)


def calculate_sins_stats(df_all):

    results = []
    for name in df_all['user'].unique():
        try:
            df = df_all[df_all['user'] == name].copy()
            df.columns = df.columns.str.lower()
            track_col = 'master_metadata_track_name'
            
            if track_col not in df.columns: continue

            if 'ts' in df.columns:
                df['ts'] = pd.to_datetime(df['ts'])
                df = df.sort_values('ts')
            
            for col in ['skipped', 'shuffle']:
                if col in df.columns: df[col] = df[col].astype(str).str.lower() == 'true'

            df = df[df[track_col].notna()].copy()

            df['change'] = df[track_col] != df[track_col].shift(1)
            df['group_id'] = df['change'].cumsum()
            sekwencje = df.groupby('group_id').agg(
                track=(track_col, 'first'),
                artist=('master_metadata_album_artist_name', 'first'),
                count=('group_id', 'count')
            )
            if not sekwencje.empty:
                max_seq = sekwencje.sort_values('count', ascending=False).iloc[0]
                rekord_petla = max_seq['count']
                rekord_info = f"<b>{max_seq['track']}</b><br>{max_seq['artist']}"
            else:
                rekord_petla, rekord_info = 0, "-"

            maniak_count = len(sekwencje[sekwencje['count'] >= 20])

            short_plays = len(df[df['ms_played'] < 20000])
            skip_ratio = (short_plays / len(df) * 100) if len(df) > 0 else 0

            fast_skips = df[(df['skipped'] == True) & (df['ms_played'] > 0)]
            if not fast_skips.empty:
                fast_ms = fast_skips['ms_played'].min()
                r = fast_skips.loc[fast_skips['ms_played'] == fast_ms].iloc[0]
                fast_info = f"<b>{r[track_col]}</b><br>{r['master_metadata_album_artist_name']}"
            else:
                fast_ms, fast_info = 0, "-" 

            manual = df['reason_start'].value_counts(normalize=True).get('clickrow', 0) * 100 if 'reason_start' in df.columns else 0

            results.append({
                'name': name,
                'petla_val': rekord_petla, 'petla_txt': rekord_info,
                'maniak_val': maniak_count,
                'skip_pct': round(skip_ratio, 2),
                'fast_val': fast_ms, 'fast_txt': fast_info,
                'manual_val': round(manual, 2), 'auto_val': round(100-manual, 2)
            })

        except Exception:
            continue

    df_final = pd.DataFrame(results)

    if not df_final.empty:
        MIN_SCORE = 20  
        MAX_SCORE = 100
        
        def normalize_column(series, invert=False):
            min_v = series.min()
            max_v = series.max()
            
            if min_v == max_v:
                return [100] * len(series)
            
            if invert:
                score = (max_v - series) / (max_v - min_v)
            else:
                score = (series - min_v) / (max_v - min_v)
            
            final_score = MIN_SCORE + (score * (MAX_SCORE - MIN_SCORE))
            return final_score

        for col in ['petla_val', 'maniak_val', 'skip_pct', 'manual_val']:
            df_final[col + '_norm'] = normalize_column(df_final[col], invert=False)
        
        real_max = df_final[df_final['fast_val'] > 0]['fast_val'].max()
        temp_series = df_final['fast_val'].replace(0, real_max if pd.notna(real_max) else 100000)
        
        df_final['fast_val_norm'] = normalize_column(temp_series, invert=True)

    return df_final


def create_radar_chart(user_data):

    categories = ['In the loop', 'Commitment issues? never.', 'Not enough dopamine?', 'Thank u, next.', 'Let Spotify cook']
    
    values = [
        user_data['petla_val_norm'], user_data['maniak_val_norm'],
        user_data['skip_pct_norm'], user_data['fast_val_norm'], user_data['manual_val_norm']]
    real_values = [
        f"Jeden utwór był najwięcej przesłuchany {user_data['petla_val']} razy!", 
        f"{user_data['maniak_val']} razy słuchano jednego utworu ponad 20 razy pod rząd!",
        f"{user_data['skip_pct']}% utworów było słuchanych przez mniej niż 20 sekund!", 
        f"Utwór został najszybciej pominięty w czasie {user_data['fast_val']} ms!", 
        f"{user_data['manual_val']}% przesłuchanych utworów był wybrany ręcznie!"]
    values += values[:1]
    categories += categories[:1]
    real_values += real_values[:1]

    fig = go.Figure(data=go.Scatterpolar(
        r=values, theta=categories, fill='toself',
        line_color=THEME['green'], fillcolor=THEME['radar_fill'], name=user_data['name'], customdata=real_values, hovertemplate='<b>%{theta}</b><br>%{customdata}<extra></extra>'))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], color=THEME['grey'], gridcolor=THEME['dark_grey'], showticklabels=False),
            angularaxis=dict(color='white', gridcolor=THEME['dark_grey'], tickfont=dict(size=16))),
        hoverlabel=dict(
            font_size=14,
            font_family="sans-serif",
            bgcolor="#2A2A2A",
            bordercolor="#1DB954",
            font_color="white"
        ),
        paper_bgcolor="#121212", plot_bgcolor="#121212",
        font=dict(color="white", size=12), showlegend=False,
        height=350, margin=dict(l=40, r=40, t=20, b=20)
    )
    return fig