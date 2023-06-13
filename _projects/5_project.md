---
layout: page
title: An interactive web map showing Valenbisi
description: Interactive map displaying public bicycles availability in Valencia
img: assets/img/valenbisi.png
importance: 2
category: fun
---

<a href="https://github.com/vitalv/valenbisi-disponibilidad"> View on GitHub</a>


<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/valenbisi.png" title="valenbisi availability map" class="img-fluid rounded z-depth-1" %}
    </div>
</div>


```python

import requests
import json
import pandas as pd
import psycopg2
import psycopg2.extras as extras
import numpy as np
```

```python

api_url = "https://valencia.opendatasoft.com/api/explore/v2.0/catalog/datasets/valenbisi-disponibilitat-valenbisi-dsiponibilidad/exports/json?limit=-1&timezone=UTC&use_labels=false&epsg=4326"

response = requests.get(api_url)

valenbisi_data = response.json()

valenbisi_df = pd.DataFrame(valenbisi_data)

valenbisi_df['num'] = valenbisi_df['number']

valenbisi_df['updated_at'] = pd.to_datetime(valenbisi_df.updated_at, format='mixed', dayfirst=True)

valenbisi_df['lat'] = [d['lat'] for d in valenbisi_df.geo_point_2d]
valenbisi_df['lon'] = [d['lon'] for d in valenbisi_df.geo_point_2d]
```


```sql
DROP DATABASE IF EXISTS valenbisi_postgis
CREATE SCHEMA valenbisi
CREATE extension postgis

CREATE TABLE valenbisi.disponibilidad (
  address VARCHAR(150),
  num INT,
  open BOOL,
  available INT,
  ticket BOOL,
  updated_at DATE,
  lat FLOAT8,
  lon FLOAT8,
  geog GEOGRAPHY(POINT, 4326)
);
```

```python
def execute_values(conn, df, table):
  
    tuples = [tuple(x) for x in df.to_numpy()]
  
    cols = ','.join(list(df.columns))
  
    # SQL query to execute
    query = "INSERT INTO %s(%s) VALUES %%s" % (table, cols)
    cursor = conn.cursor()
    try:
        extras.execute_values(cursor, query, tuples)
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error: %s" % error)
        conn.rollback()
        cursor.close()
        return 1
    print("execute_values() done")
    cursor.close()
  
  
# establishing connection
conn = psycopg2.connect(
    database="valenbisi_postgis",
    user='postgres',
    password='chunkybacon',
    host='127.0.0.1',
    port='5432'
)

cursor = conn.cursor()

zip_lon_lat = list(zip(valenbisi_df.lon, valenbisi_df.lat))

valenbisi_df['geog'] = [f'POINT({lon_lat[0]} {lon_lat[1]})' for lon_lat in zip_lon_lat]

data = valenbisi_df[["address", "num", "open", "available", "ticket", "updated_at", "lat", "lon", "geog"]]

execute_values(conn, data, 'valenbisi.disponibilidad')
```

```sql
ALTER TABLE valenbisi.disponibilidad RENAME COLUMN geog TO geom;

ALTER TABLE 
  valenbisi.disponibilidad ALTER COLUMN geom TYPE geometry(POINT, 3395)
  USING ST_Transform(geom::geometry, 3395);
```


