import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
from sklearn.preprocessing import MinMaxScaler
import numpy as np

PLIKI_KONFIGURACJA = [
    {'path': 'dane_basia.csv', 'name': 'Basia'},
    {'path': 'dane_adam.csv', 'name': 'Adam'},
    {'path': 'dane_pawel.csv', 'name': 'Paweł'}
]

st.set_page_config(page_title="Spotify Stats Pro", page_icon="🎧", layout="wide")

st.markdown("""
    <style>
        .stApp { background-color: #121212; color: white; }
        h1, h2, h3 { color: #1DB954 !important; font-family: sans-serif; }
        button[data-baseweb="tab"] { background-color: transparent !important; color: #B3B3B3 !important; border-bottom: 2px solid #333 !important; }
        button[data-baseweb="tab"][aria-selected="true"] { color: #1DB954 !important; border-bottom: 2px solid #1DB954 !important; }
    </style>
""", unsafe_allow_html=True)

THEME = {
    'green': '#1DB954', 'red': "#AC0F1F", 'blue': '#2E77D0',
    'yellow': '#F59B23', 'grey': '#535353', 'dark_grey': '#2A2A2A', 'pink': "#700B6B",
    'radar_fill': 'rgba(29, 185, 84, 0.4)'
}

@st.cache_data
def load_data(file_config):
    results = []
    
    for entry in file_config:
        path = entry['path']
        name = entry['name']
        
        if not os.path.exists(path): continue
            
        try:
            df = pd.read_csv(path)
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
    categories = ['In the loop', 'ADHD check!', 'Not enough dopamine?', 'Thank u, next.', 'Let Spotify cook']
    values = [
        user_data['petla_val_norm'], user_data['maniak_val_norm'],
        user_data['skip_pct_norm'], user_data['fast_val_norm'], user_data['manual_val_norm']
    ]
    values += values[:1]
    categories += categories[:1]

    fig = go.Figure(data=go.Scatterpolar(
        r=values, theta=categories, fill='toself',
        line_color=THEME['green'], fillcolor=THEME['radar_fill'], name=user_data['name']
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], color=THEME['grey'], gridcolor=THEME['dark_grey']),
            angularaxis=dict(color='white', gridcolor=THEME['dark_grey'])
        ),
        paper_bgcolor="#121212", plot_bgcolor="#121212",
        font=dict(color="white", size=12), showlegend=False,
        height=350, margin=dict(l=40, r=40, t=20, b=20)
    )
    return fig

st.title("Muzyczne grzechy")
df_res = load_data(PLIKI_KONFIGURACJA)

st.divider()
col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("Wybierz kategorię:")
    kategorie = {
        "🎵 In the loop": "loop",
        "💀 ADHD check!": "maniac",
        "⏩ Not enough dopamine?": "impatience",
        "❌ Thank u, next.": "speed",
        "🔥 Let Spotify cook": "control"
    }
    sel = st.radio("x", list(kategorie.keys()), label_visibility="collapsed")
    mode = kategorie[sel]

with col2:
    fig = go.Figure()
    if mode == "loop":
        fig.add_trace(go.Bar(x=df_res['name'], y=df_res['petla_val'], textfont=dict(size=15), text=df_res['petla_val'], textposition='auto', hovertext=df_res['petla_txt'], hoverinfo="text", marker_color=THEME['green']))
        t = "Najwięcej odsłuchań tej samej piosenki pod rząd"
    elif mode == "maniac":
        fig.add_trace(go.Bar(x=df_res['name'], y=df_res['maniak_val'], textfont=dict(size=15), text=df_res['maniak_val'], textposition='auto', marker_color=THEME['green']))
        t = "Ile razy odtwarzano jeden utwór ponad 20 razy pod rząd?"
    elif mode == "impatience":
        fig.add_trace(go.Bar(x=df_res['name'], y=df_res['skip_pct'], textfont=dict(size=15), text=df_res['skip_pct'].apply(lambda x: f"{x}%"), textposition='auto', marker_color=THEME['green']))
        t = "Jaki procent słuchanych utworów odtwarzaliśmy przez mniej niż 20 sekund?"
    elif mode == "speed":
        fig.add_trace(go.Bar(x=df_res['name'], y=df_res['fast_val'], textfont=dict(size=15), text=df_res['fast_val'].apply(lambda x: f"{x} ms"), textposition='auto', hovertext=df_res['fast_txt'], hoverinfo="text", marker_color=THEME['green']))
        t = "Najszybsze pominięcie utworu"
    elif mode == "control":
        fig.add_trace(go.Bar(x=df_res['name'], y=df_res['manual_val'], textfont=dict(size=15), text=df_res['manual_val'].apply(lambda x: f"{x:.1f}%"), textposition='auto', marker_color=THEME['green'], name='Wybór ręczny'))
        fig.add_trace(go.Bar(x=df_res['name'], y=df_res['auto_val'], textfont=dict(size=15), marker_color=THEME['red'], name='Wybór spotify'))
        t = "Jaki procent słuchanych utworów wybieraliśmy ręcznie?"

    fig.update_layout(title=t, paper_bgcolor="#121212", plot_bgcolor="#121212", font=dict(color="white"), showlegend=False, barmode='stack', height=400, margin=dict(l=20, r=20, t=40, b=20), xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#333"))
    st.plotly_chart(fig, use_container_width=True)

st.header("Jak wypadamy na tle reszty?")
st.divider()
tabs = st.tabs(df_res['name'].tolist())

OPISY = {
    'Basia': 'Attention span shorter than a goldfish ☠️',
    'Adam': 'The normal one❓❓❓',
    'Paweł': 'My ADHD is on steroids🐳',
}

for i, tab in enumerate(tabs):
    with tab:
        u = df_res.iloc[i]       
        opis = OPISY.get(u['name'])
        c1, c2 = st.columns([3, 2]) 
        with c1: 
            st.plotly_chart(create_radar_chart(u), use_container_width=True)
        with c2:
            st.subheader(f"{u['name']}")
            st.markdown(f"""
                <div style='color: #1DB954; font-size: 1.1em; font-style: italic; margin-bottom: 20px;'>
                    {opis}
                </div>
            """, unsafe_allow_html=True)
            st.write(f"Jeden utwór był najwięcej przesłuchany **{u['petla_val']}** razy pod\u00A0rząd!")
            st.write(f"**{u['maniak_val']}** razy słuchano jednego utworu ponad 20 razy pod\u00A0rząd!")
            st.write(f"**{u['skip_pct']}%** utworów było słuchanych przez mniej niż 20 sekund!")
            st.write(f"Utwór został najszybciej pominięty w czasie **{u['fast_val']} milisekund**!")
            st.write(f"**{u['manual_val']}%** przesłuchanych utworów był wybrany ręcznie!")