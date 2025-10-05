import streamlit as st
import math
import gpxpy
import gpxpy.gpx
import requests
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

st.set_page_config(page_title="Prédiction course - GPX + météo (forcée)", layout="wide")
st.title("🏃‍♂️ Prédiction de course avec GPX, distance forcée et météo horaire")

# ---------------- utilitaires ----------------
def hms_to_seconds(hms: str) -> int:
    h, m, s = map(int, hms.split(":"))
    return h*3600 + m*60 + s

def seconds_to_hms(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}"

def parse_gpx_points(file):
    gpx = gpxpy.parse(file)
    points = []
    for track in gpx.tracks:
        for segment in track.segments:
            for p in segment.points:
                points.append(p)
    return gpx, points

def gpx_cumulative_distance_and_elev(points):
    cum_d = [0.0]
    elevs = [points[0].elevation if points[0].elevation is not None else 0.0]
    total = 0.0
    for i in range(1, len(points)):
        d = points[i].distance_3d(points[i-1])
        total += d
        cum_d.append(total)
        elevs.append(points[i].elevation if points[i].elevation is not None else elevs[-1])
    return cum_d, elevs, total

def interp_elevation_at(dist_target, cum_d, elevs):
    # if target beyond last point, return last elevation
    if dist_target <= 0:
        return elevs[0]
    if dist_target >= cum_d[-1]:
        return elevs[-1]
    # find segment
    for i in range(1, len(cum_d)):
        if cum_d[i] >= dist_target:
            d0 = cum_d[i-1]
            d1 = cum_d[i]
            e0 = elevs[i-1]
            e1 = elevs[i]
            if d1 == d0:
                return e1
            frac = (dist_target - d0) / (d1 - d0)
            return e0 + frac * (e1 - e0)
    return elevs[-1]

@st.cache_data(ttl=600)
def fetch_weather_forecast(api_key, lat, lon):
    """Récupère la prévision horaire (onecall / forecast)."""
    if not api_key:
        return None
    # On utilise OneCall (forecast) endpoint (hourly). OpenWeather may require paid plan for some endpoints;
    # fallback to 'forecast' 3h if needed.
    try:
        url = f"https://api.openweathermap.org/data/2.5/onecall?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            # prefer hourly if present
            if "hourly" in data:
                return {"type": "onecall", "data": data["hourly"], "tz_offset": data.get("timezone_offset", 0)}
        # fallback to 3-hour forecast
        url2 = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        r2 = requests.get(url2, timeout=10)
        if r2.status_code == 200:
            data2 = r2.json()
            return {"type": "forecast3h", "data": data2.get("list", []), "tz_offset": 0}
    except Exception:
        return None
    return None

def find_closest_weather_entry(weather_cache, target_dt):
    """Donne la meilleure entrée météo (temp, wind) pour target_dt."""
    if weather_cache is None:
        return None
    entries = weather_cache["data"]
    # entries contain 'dt' unix timestamps
    best = min(entries, key=lambda x: abs(datetime.fromtimestamp(x["dt"]) - target_dt))
    # structure differs slightly between endpoints
    temp = best.get("temp") or best.get("main", {}).get("temp")
    wind = best.get("wind", {}).get("speed") if best.get("wind") else best.get("wind_speed") if best.get("wind_speed") else None
    if wind is None:
        # try top-level 'wind' dict
        wind = best.get("wind", {}).get("speed", 0)
    return {"temp": temp, "wind": wind}

# ---------------- UI ----------------
st.markdown("### 1) Upload GPX (track) — 1 point par km sera extrait via interpolation")
gpx_file = st.file_uploader("Fichier GPX (track recommandé)", type=["gpx"])

st.markdown("### 2) Paramètres course & régression (3 courses de référence)")
courses = []
for i in range(1,4):
    c0, c1, c2, c3 = st.columns([1.2,1.2,1,1])
    with c0:
        dist_i = st.number_input(f"Dist {i} (m)", min_value=1, step=100, value=5000*i, key=f"dist{i}")
    with c1:
        temps_i = st.text_input(f"Temps {i} (h:mm:ss)", value=f"0:{40+i*2}:00", key=f"temps{i}")
    with c2:
        up_i = st.number_input(f"D+ {i} (m)", min_value=0, step=1, value=0, key=f"up{i}")
    with c3:
        down_i = st.number_input(f"D- {i} (m)", min_value=0, step=1, value=0, key=f"down{i}")
    courses.append({"distance": dist_i, "temps": temps_i, "D_up": up_i, "D_down": down_i})

st.markdown("### 3) Coefficients (modifiable)")
c1, c2 = st.columns(2)
with c1:
    k_up = st.number_input("k_montée (exponentiel)", min_value=1.0, max_value=2.0, step=0.00001, value=1.00100, format="%.5f")
    k_down = st.number_input("k_descente", min_value=0.90, max_value=1.0, step=0.00001, value=0.99900, format="%.5f")
with c2:
    k_temp_sup = st.number_input("k_temp_sup (>20°C)", min_value=0.90, max_value=1.1, step=0.00001, value=1.00200, format="%.5f")
    k_temp_inf = st.number_input("k_temp_inf (<20°C)", min_value=0.90, max_value=1.1, step=0.00001, value=0.99800, format="%.5f")

st.markdown("### 4) Distance officielle & météo")
cold1, cold2, cold3 = st.columns([1.2,1.2,1])
with cold1:
    distance_officielle = st.number_input("Distance officielle (m) — forcer la distance (ex: 42195)", min_value=1000, step=1, value=42195)
with cold2:
    date_course = st.date_input("Date de la course")
    heure_course = st.time_input("Heure de départ")
with cold3:
    latitude = st.number_input("Latitude", value=48.8566)
    longitude = st.number_input("Longitude", value=2.3522)
    API_KEY = st.text_input("Clé API OpenWeather (optionnel)", type="password")

# bouton
if st.button("Lancer analyse & calcul dynamique"):
    if gpx_file is None:
        st.error("Upload un fichier GPX d'abord.")
        st.stop()

    try:
        # parse gpx and diagnostics
        gpx, points = parse_gpx_points(gpx_file)
        st.write(f"Tracks: {len(gpx.tracks)} | Segments: {sum(len(t.segments) for t in gpx.tracks)} | Points: {len(points)}")
        cum_d, elevs, total_len_m = gpx_cumulative_distance_and_elev(points)
        st.write(f"Distance GPX calculée = {total_len_m:.1f} m  ({total_len_m/1000:.3f} km)")

        # determine number of km bins from official distance (cap to 42)
        n_km = int(distance_officielle // 1000)
        if n_km > 42:
            n_km = 42
        st.write(f"Analyse forcée sur {n_km} km (distance officielle demandée = {distance_officielle} m)")

        # build elevation at each km mark (0..n_km)
        km_markers = [i * (distance_officielle / n_km) for i in range(0, n_km+1)]  # distribute official distance evenly
        # But better: for per-km markers use exact multiples of 1000 if official distance multiple of 1000,
        # here we want one point per km, so we use exact multiples of 1000 up to n_km
        km_marks_exact = [i * 1000 for i in range(0, n_km+1)]

        # interpolate elevations at km marks using GPX cumulative distances/elevs
        elev_at_km = []
        for km_dist in km_marks_exact:
            # if official distance > GPX total, clamp km_dist to last point
            target = min(km_dist, cum_d[-1])
            elev_at_km.append(interp_elevation_at(target, cum_d, elevs))

        # compute D+ / D- per km between consecutive km marks
        per_km = []
        for i in range(1, len(elev_at_km)):
            delta_h = elev_at_km[i] - elev_at_km[i-1]
            d_up = max(0.0, delta_h)
            d_down = max(0.0, -delta_h)
            per_km.append({"D_up": d_up, "D_down": d_down, "km_index": i})

        # If actual GPX total is much shorter than official distance, warn user
        if total_len_m < distance_officielle * 0.9:
            st.warning("Attention : la distance GPX est beaucoup plus courte que la distance officielle. "
                       "Les altitudes seront extrapolées depuis le dernier point disponible.")

        # Regression log-log using past courses (apply denivele adjustments as before)
        temps_sec = []
        dists_ref = []
        for c in courses:
            t = hms_to_seconds(c["temps"])
            t_adj = t * (k_up ** c["D_up"]) * (k_down ** c["D_down"])
            temps_sec.append(t_adj)
            dists_ref.append(c["distance"])

        # compute K_pred using all pairs
        K_list = []
        for i in range(len(temps_sec)):
            for j in range(i+1, len(temps_sec)):
                if temps_sec[i] > 0 and dists_ref[i] > 0 and temps_sec[j] > 0 and dists_ref[j] > 0:
                    K_ij = math.log(temps_sec[j]/temps_sec[i]) / math.log(dists_ref[j]/dists_ref[i])
                    K_list.append(K_ij)
        if len(K_list) == 0:
            st.error("Impossible d'estimer K (vérifie les courses de référence).")
            st.stop()
        K_pred = sum(K_list) / len(K_list)
        st.info(f"Exposant log-log K estimé = {K_pred:.4f}")

        # base predicted total time scaled to official distance (use last ref as base)
        base_time_total = temps_sec[-1] * (distance_officielle / dists_ref[-1])**K_pred
        base_s_per_km = base_time_total / (distance_officielle/1000.0)

        # get weather forecast once
        weather_cache = fetch_weather_forecast(API_KEY, latitude, longitude)
        dt_depart = datetime.combine(date_course, heure_course)

        # compute per-km times applying denivele + weather factors
        km_results = []
        cum_time = 0.0
        for idx, km in enumerate(per_km):
            idx_k = idx + 1
            length_km = 1.0  # 1 km
            t_km = base_s_per_km * length_km
            # denivele effect
            t_km *= (k_up ** km["D_up"]) * (k_down ** km["D_down"])
            # estimate passage time for this km (use cum_time + t_km)
            passage_dt = dt_depart + timedelta(seconds=(cum_time + t_km))
            # weather for passage hour
            weather = find_closest_weather_entry(weather_cache, passage_dt) if weather_cache else None
            temp = weather["temp"] if weather and weather.get("temp") is not None else 20.0
            wind = weather["wind"] if weather and weather.get("wind") is not None else 0.0
            # temperature adjustment
            if temp > 20:
                t_km *= (k_temp_sup ** (temp - 20))
            else:
                t_km *= (k_temp_inf ** (20 - temp))
            cum_time += t_km
            pace_min = int((t_km/60)//1)
            pace_sec = int((t_km/60 - pace_min) * 60)
            slope_pct = (km["D_up"] - km["D_down"]) / 1000.0 * 100.0  # approx slope %
            km_results.append({
                "Km": idx_k,
                "D+ (m)": round(km["D_up"],1),
                "D- (m)": round(km["D_down"],1),
                "Slope (%)": round(slope_pct,3),
                "Heure prévue": passage_dt.strftime("%H:%M"),
                "Temp (°C)": round(temp,1),
                "Vent (m/s)": round(wind,1),
                "Temps segment (s)": round(t_km,1),
                "Allure (min/km)": f"{int(t_km//60)}:{int(t_km%60):02d}"
            })

        total_sec = sum(r["Temps segment (s)"] for r in km_results)
        st.subheader("Résultats synthèse")
        st.write(f"Temps total prévisionnel (forcé) = {seconds_to_hms(total_sec)}")
        avg_pace = total_sec / (distance_officielle/1000.0)
        st.write(f"Allure moyenne estimée = {int(avg_pace//60)}:{int(avg_pace%60):02d} min/km")

        # ---------- affichage graphique (largeur) ----------
        st.subheader("Graphiques : profil & allure vs pente")
        # prepare plotting arrays
        km_idxs = [r["Km"] for r in km_results]
        elevs_km = [interp_elevation_at(min(k*1000, cum_d[-1]), cum_d, elevs) for k in km_idxs]
        slopes = [r["Slope (%)"] for r in km_results]
        paces_sec = [r["Temps segment (s)"] for r in km_results]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,4), sharex=True)
        # elevation profile
        ax1.plot(km_idxs, elevs_km, marker='o')
        ax1.set_xlabel("Km")
        ax1.set_ylabel("Altitude (m)")
        ax1.set_title("Profil d'altitude (interpolé par km)")
        ax1.grid(True)
        # pace vs slope
        # convert pace to min/km for plotting
        paces_minpkm = [p/60.0 for p in paces_sec]
        ax2.scatter(slopes, paces_minpkm)
        ax2.plot(slopes, paces_minpkm, alpha=0.3)
        ax2.set_xlabel("Pente approximative (%)")
        ax2.set_ylabel("Allure (min/km)")
        ax2.set_title("Allure vs pente (par km)")
        ax2.grid(True)

        st.pyplot(fig)

        # show dataframe (km table)
        st.subheader("Tableau km par km")
        st.dataframe(km_results)

    except Exception as e:
        st.error(f"Erreur durant l'analyse : {e}")
