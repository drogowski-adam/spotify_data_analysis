import pandas as pd

FILES = {
    "Adam": "dane/dane_adam.csv",
    "Basia": "dane/dane_basia.csv",
    "Paweł": "dane/dane_pawel.csv"
}

wyniki = []

for user, path in FILES.items():
    df = pd.read_csv(path)

    df = df.dropna(subset=["master_metadata_album_artist_name"])

    top_artists = (
        df
        .groupby("master_metadata_album_artist_name")["ms_played"]
        .sum()
        .reset_index(name="total_ms")
        .sort_values("total_ms", ascending=False)
        .head(5)
    )

    for _, row in top_artists.iterrows():
        minutes = row["total_ms"] / 60000
        hours = minutes / 60
        
        wyniki.append({
            "user": user,
            "artist": row["master_metadata_album_artist_name"],
            "minutes_listened": round(minutes, 1),
            "hours_listened": round(hours, 2)
        })

result_df = pd.DataFrame(wyniki)
result_df.to_csv("dane/our_top_15_artists.csv", index=False, encoding="utf-8")

print("Gotowe! Zapisano plik: dane/our_top_15_artists.csv")