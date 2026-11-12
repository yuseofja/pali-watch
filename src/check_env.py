import hdx.api, geopandas, plotly, folium, streamlit, pandas, requests
print("geopandas", geopandas.__version__)
print("streamlit", streamlit.__version__)
print("All good.")


import pandas as pd
from hdx.api.configuration import Configuration
from hdx.data.dataset import Dataset

Configuration.create(
    hdx_site="prod",
    user_agent="pali-dashboard-research",
    hdx_read_only=True,
)

datasets = Dataset.search_in_hdx("groups:pse", rows=500)

rows = []
for d in datasets:
    rows.append({
        "title": d["title"],
        "name": d["name"],
        "url": f"https://data.humdata.org/dataset/{d['name']}",
        "last_modified": d.get("last_modified"),
        #"formats": ", ".join(sorted(set(d.get_formats()))),
        "source": d.get("dataset_source"),
        "notes": (d.get("notes") or "")[:200],
    })

df = pd.DataFrame(rows).sort_values("last_modified", ascending=False)
df.to_csv("data/processed/hdx_pse_catalog_raw.csv", index=False)
print(f"{len(df)} datasets found")



LEFT OFF HERE:
        - focus on narrow scope of 2 week sprint: what datasets are useful?
        - save this dataset pulling script so I can come back to this later 
        - document that the script exists and what it does? (maybe save this for next sprint, don't get bogged down by infrastructure rn)
        - pick datasets to start with                                                    
        