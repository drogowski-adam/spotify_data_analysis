import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import random
from datetime import datetime

from helper_funcs import (
    load_main_data,
    load_quotes,
    load_mood_df,
    hour_agg,
    render_dashboard,
    calculate_sins_stats,
    create_radar_chart,
    MOOD_FILES
)

from app_adam import (
    load_df_adam,
    hour_agg_adam,
    render_dashboard_adam
)

from theme import THEME


st.set_page_config(
    page_title="Taking Spotify off the record",
    page_icon="🎵",
    layout="wide"
)


st.markdown("""
    <style>
        .stApp { background-color: #121212; color: white; }
        h1, h2, h3 { color: #1DB954 !important; font-family: sans-serif; }
        button[data-baseweb="tab"] { background-color: transparent !important; color: #B3B3B3 !important; border-bottom: 2px solid #333 !important; }
        button[data-baseweb="tab"][aria-selected="true"] { color: #1DB954 !important; border-bottom: 2px solid #1DB954 !important; }
    </style>
""", unsafe_allow_html=True)


df_all = load_main_data()
quotes_df = load_quotes()

if df_all.empty:
    st.error("Błąd krytyczny: Brak głównych plików CSV (dane_adam.csv itp).")
    st.stop()


with st.sidebar:

    st.title("🎵 Taking Spotify off the record")
    users = ["Adam", "Paweł", "Basia"]
    selected_user = st.selectbox("Wybierz osobę:", users)
    
    if st.session_state.get("last_game_user") != selected_user:
        st.session_state.game_result = None
        st.session_state.last_game_user = selected_user
        keys_to_remove = [k for k in st.session_state.keys() if k.startswith("game_opts_")]
        for k in keys_to_remove:
            del st.session_state[k]


df = df_all[df_all['user'] == selected_user]

tab0, tab1, tab2, tab3, tab4 = st.tabs(
    ["O projekcie", "Jesteśmy przewidywalni", "Wydział beznadziejności", "Dni i noce", "Tragiczne cytaty"]
)


with tab0:

    st.header("Witaj w naszej muzycznej kryjówce!")
    st.markdown("""
    Dashboard został stworzony w ramach projektu z Technik Wizualizacji Danych. 
    Analizujemy w nim nasze historie aktywności ze Spotify, szukając dziwactw, obsesji i szreoko pojętych **najgorszych muzycznych nawyków**.

    ### Co znajdziesz w środku?

    * **Jesteśmy przewidywalni** – Gra w zgadywanie artysty. Algorytm sprawdza, czego słuchaliśmy o tej konkretnej godzinie w ciągu ostatniego roku i każe Ci zgadywać.
    * **Wydział beznadziejności** – Porównanie naszych najgorszych nawyków: obsesyjne pętle, pomijanie utworów po 2 sekundach i maniakalne słuchanie tego samego utworu. Wykresy radarowe pokazują, kto jest największym "grzesznikiem".
    * **Dni i noce** – Heatmapy pokazujące, jak zmienia się nastrój, taneczność i melancholiność słuchanej przez nas muzyki w zależności od pory dnia.
    * **Tragiczne cytaty** – Quiz, w którym musisz rozpoznać, czy wyświetlony fatalny cytat z piosenki należy do tekstu jednego z ulubionych utworów wybranej osoby, czy może któregoś innego z nas.
    
    👈 **Wybierz osobę z panelu bocznego, aby rozpocząć!**
    """)


with tab1:

    st.header("Czego byśmy teraz słuchali?")
    current_hour = datetime.now().hour
    st.caption(f"Godzina: {current_hour}:00 | Osoba: {selected_user}")

    max_date = df_all['ts'].max()
    cutoff_date = max_date - pd.DateOffset(years=1)
    df_last_year = df[df['ts'] > cutoff_date]
    df_hour = df_last_year[df_last_year['hour'] == current_hour]
    
    if df_hour.empty:
        st.warning("Brak danych o tej godzinie.")
    else:
        correct_artist = df_hour['master_metadata_album_artist_name'].value_counts().idxmax()
        
        game_key = f"game_opts_{selected_user}_{current_hour}"
        
        if game_key not in st.session_state:
            all_artists = df['master_metadata_album_artist_name'].unique().tolist()
            pool = [a for a in all_artists if a != correct_artist]
            k = min(2, len(pool))
            distractors = random.sample(pool, k=k) if k > 0 else []
            options = distractors + [correct_artist]
            random.shuffle(options)
            st.session_state[game_key] = options
            st.session_state.game_result = None

        options = st.session_state[game_key]

        cols = st.columns(3)
        for i, option in enumerate(options):
            if cols[i].button(option, key=f"btn_{selected_user}_{i}", use_container_width=True):
                if option == correct_artist:
                    st.session_state.game_result = True
                else:
                    st.session_state.game_result = False

        if st.session_state.game_result is True:
            st.balloons()
            st.success(f"TRAFIONE! **{selected_user}** najpewniej słuchał(a)by: **{correct_artist}**")
        elif st.session_state.game_result is False:
            st.error(f"NIE TYM RAZEM! Dane sugerowały: **{correct_artist}**")


with tab2:
    
    df_res = calculate_sins_stats(df_all)

    st.header("Nikt nie pytał, a każdy chciałby wiedzieć")
    st.markdown("Oto dogłębne stadium przypadków beznadziejnych. Sprawdź, jak bardzo dziwni jesteśmy. Po prostu.")
    st.divider()
    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Wybierz kategorię:")
        kategorie = {
            "🎵 In the loop": "loop",
            "🤝 Commitment issues? never.": "maniac",
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
    st.markdown("Poniżej znajduje się porównanie naszych cech na wspólnej skali. Im punkt dalej od środka, tym bardziej odstajemy w danej kategorii od reszty. Zewnętrzna krawędź reprezentuje największy odnotowany wynik.")
    st.markdown("Najedź na wierzchołki zielonego pola, by zobaczyć szczegółowe dane.")
    st.divider()
    
    tabs_sins = st.tabs(df_res['name'].tolist())

    OPISY = {
        'Basia': 'Attention span shorter than a goldfish ☠️',
        'Adam': 'The normal one❓❓❓',
        'Paweł': 'Hyperfixation at its finest🐳',
    }
    for i, tab in enumerate(tabs_sins):
        with tab:
            u = df_res.iloc[i]       
            opis = OPISY.get(u['name'], "")
            
            st.markdown(f"""
                <div style="text-align: center;">
                    <h2 style="color: white; margin-bottom: 5px;">{u['name']}</h2>
                    <div style="color: #1DB954; font-size: 1.2em; font-style: italic; margin-bottom: 10px;">
                        {opis}
                    </div>
                </div>
            """, unsafe_allow_html=True)
            st.plotly_chart(create_radar_chart(u), use_container_width=True)

with tab3:

    st.header("Czym różni się dzień od nocy?")
    st.caption(f"Profil dobowy — **{selected_user}**")

    from app_adam import FILES_DAY_NIGHT

    df_dn = load_df_adam(FILES_DAY_NIGHT[selected_user])

    months = sorted(df_dn["month"].dropna().unique())
    if not months:
        st.warning("Brak danych miesięcznych.")
        st.stop()

    c1, c2 = st.columns(2)
    start_month = c1.selectbox("Od miesiąca", months, index=0)
    end_month = c2.selectbox("Do miesiąca", months, index=len(months) - 1)

    if start_month > end_month:
        start_month, end_month = end_month, start_month

    df_sel = df_dn[(df_dn["month"] >= start_month) & (df_dn["month"] <= end_month)]

    agg = hour_agg_adam(df_sel)

    render_dashboard_adam(agg, selected_user, start_month, end_month)

    #st.header("Czym różni się dzień od nocy?")
    #st.caption(f"Profil dobowy — **{selected_user}**")
    
    #file_path = MOOD_FILES.get(selected_user)
    #df_mood = load_mood_df(file_path)

    #if df_mood.empty:
        #st.error(f"⚠️ Nie znaleziono pliku `{file_path}` w folderze `dane/`.")
    #else:
        #months = sorted(df_mood["month"].dropna().unique())
        
        #if len(months) == 0:
            #st.warning("Plik załadowany, ale brak danych czasowych.")
        #else:
            #c1, c2 = st.columns(2)
            #start_m = c1.selectbox("Od:", months, index=0)
            #end_m = c2.selectbox("Do:", months, index=len(months)-1)

            #if start_m > end_m:
                #st.error("Data końcowa musi być późniejsza niż początkowa.")
            #else:
                #mask = (df_mood["month"] >= start_m) & (df_mood["month"] <= end_m)
                #df_sel = df_mood[mask]
                
                #if df_sel.empty:
                    #st.warning("Brak danych w wybranym okresie.")
                #else:
                    #agg = hour_agg(df_sel)
                    #st.markdown(f"**Analiza:** {len(df_sel)} odtworzeń ({int(df_sel['minutes'].sum())} minut)")
                    
                    #render_dashboard(agg, selected_user, start_m, end_m)


with tab4:

    st.header("Kto tego może w ogóle słuchać???")
    st.markdown("Poniżej zobaczysz 3 okropne cytaty. **Zgadnij, który z nich pochodzi z jednego z ulubionych utworów obecnie wybranej osoby.**")

    quote_key = f"quote_opts_{selected_user}"

    if st.button("Losuj nowe rozdanie"):
        if quote_key in st.session_state:
            del st.session_state[quote_key]
        st.session_state.quote_result = None

    if quote_key not in st.session_state:

        target_quotes = quotes_df[quotes_df['user'] == selected_user]
        
        other_quotes = quotes_df[quotes_df['user'] != selected_user]
        
        if target_quotes.empty or len(other_quotes) < 2:
            st.warning("Za mało danych w pliku z cytatami, żeby zagrać.")
            game_ready = False

        else:
            correct_row = target_quotes.sample(1)
            distractor_rows = other_quotes.sample(2)
            game_df = pd.concat([correct_row, distractor_rows]).sample(frac=1).reset_index(drop=True)
            
            st.session_state[quote_key] = {
                "data": game_df,
                "correct_track": correct_row.iloc[0]['track']
            }
            game_ready = True

    else:

        game_ready = True

    if game_ready:

        game_data = st.session_state[quote_key]["data"]
        correct_track = st.session_state[quote_key]["correct_track"]

        cols = st.columns(3)
        
        for i, row in game_data.iterrows():

            with cols[i]:
                st.markdown(
                    f"""
                    <div style="border:1px solid #333; padding: 20px; border-radius: 10px; height: 250px; display: flex; flex-direction: column; justify-content: space-between;">
                        <div style="font-size: 18px; font-style: italic; color: #E0E0E0;">
                            "{row['quote']}"
                        </div>
                        <div style="margin-top: 15px; color: #1DB954; font-weight: bold;">
                            {row['artist']} <br> <span style="color: #999; font-weight: normal;">{row['track']}</span>
                        </div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                if st.button(f"To {selected_user}!", key=f"q_btn_{i}", use_container_width=True):
                    if row['track'] == correct_track:
                        st.session_state.quote_result = True
                    else:
                        st.session_state.quote_result = False

        if st.session_state.get("quote_result") is True:
            st.balloons()
            st.success(f"TAK! Ten ambitny tekst należy do historii słuchania osoby: **{selected_user}**.")

        elif st.session_state.get("quote_result") is False:
            st.error(f"NIE! To akurat tekst kogoś innego. Spróbuj zgadnąć ponownie.")