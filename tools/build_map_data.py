#!/usr/bin/env python3
"""
Build map-data.json — the vector-occurrence layer shown on the MosquitoNet map.

WHY THIS EXISTS
  The map used to be generated in the browser by scattering random dots around
  ~255 city centres. That produced three classes of error:
    * straight diagonal lines — the fallback used the SAME random offset for lat
      and lng, and crude "ocean box" rectangles (which wrongly covered India,
      Brazil, Mexico, Central America, Tokyo, Houston, Chicago, LA...) sent 75
      cities down that path;
    * dots at sea — up to 13-39 km of random scatter around coastal cities;
    * fake precision — every dot was synthetic; none was an observed record.
  This script replaces that with records that are each a real, documented
  location, validated against a 1:10m land mask.

SOURCES
  1. CURATED  — city-level documented vector presence (below). Each row is one
     place where the taxon is documented; no invented counts, no scatter.
  2. GBIF     — real geo-referenced occurrence records (includes the Kraemer et
     al. 2015 global Aedes compendium, doi:10.15468/bgmqmr / 10.15468/7apj8n).
     Only fetched with --gbif, and requires network access to api.gbif.org.

CLEANING (applied to every record, curated or fetched)
  * range / (0,0) / non-numeric checks
  * land test against Natural Earth 1:10m land + minor islands, minus lakes;
    points within SNAP_KM of land (harbour/estuary centroids) are snapped to the
    nearest coast, anything further out is DROPPED
  * de-duplication per species on a ~1 km grid (0.01 deg)
  * GBIF only: coordinateUncertainty <= 10 km, no geospatial issues, year >= 1990,
    thinned to one record per species per 0.1 deg cell (keeps the map fast)

USAGE
  python3 tools/build_map_data.py            # curated only
  python3 tools/build_map_data.py --gbif     # + GBIF occurrences (needs network)
"""
import json, math, os, sys, time, urllib.request, urllib.parse
from datetime import date

ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, 'tools', '.cache')
OUT   = os.path.join(ROOT, 'map-data.json')
NE    = 'https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/'
SNAP_KM = 3.0

SOURCES = {
  'curated': 'City-level documented presence compiled from WHO World Malaria Report 2025, '
             'PAHO dengue bulletins 2024, ECDC invasive-mosquito maps (Jun 2025), CDC ArboNET, '
             'Kraemer et al. 2015 eLife (doi:10.7554/eLife.08347), Sinka et al. 2012 '
             '(dominant Anopheles vectors), and An. stephensi first-detection reports '
             '(Carter et al. 2018; Kenya/Ghana EID 2023-24; Yemen 2021-22; Sri Lanka 2017).',
  'gbif':    'GBIF.org occurrence records (CC0/CC-BY), incl. Kraemer et al. 2015 global '
             'compendium of Ae. aegypti and Ae. albopictus occurrence.',
}

# (place, lat, lng, app species id, year reported/first documented, status, taxon)
CURATED = [
    ('Lagos', 6.524, 3.379, 'anopheles', 2024, 'established', 'An. gambiae s.l. / An. funestus'),
    ('Abuja', 9.072, 7.491, 'anopheles', 2024, 'established', 'An. gambiae s.l. / An. funestus'),
    ('Kano', 12.002, 8.592, 'anopheles', 2024, 'established', 'An. gambiae s.l. / An. funestus'),
    ('Ibadan', 7.376, 3.947, 'anopheles', 2024, 'established', 'An. gambiae s.l. / An. funestus'),
    ('Benin City', 6.335, 5.627, 'anopheles', 2024, 'established', 'An. gambiae s.l. / An. funestus'),
    ('Port Harcourt', 4.815, 7.049, 'anopheles', 2024, 'established', 'An. gambiae s.l. / An. funestus'),
    ('Ilorin', 8.492, 4.541, 'anopheles', 2024, 'established', 'An. gambiae s.l. / An. funestus'),
    ('Maiduguri', 10.298, 13.283, 'anopheles', 2023, 'established', 'An. gambiae s.l. / An. funestus'),
    ('Kinshasa', -4.322, 15.322, 'anopheles', 2024, 'established', 'An. gambiae s.l. / An. funestus'),
    ('Goma', -1.679, 29.222, 'anopheles', 2024, 'established', 'An. gambiae s.l. / An. funestus'),
    ('Kisangani', 0.517, 25.198, 'anopheles', 2023, 'established', 'An. gambiae s.l. / An. funestus'),
    ('Lubumbashi', -11.663, 27.479, 'anopheles', 2023, 'established', 'An. gambiae s.l. / An. funestus'),
    ('Niamey', 13.512, 2.125, 'anopheles', 2024, 'established', 'An. gambiae s.l. / An. funestus'),
    ('Zinder', 13.808, 8.988, 'anopheles', 2023, 'established', 'An. gambiae s.l. / An. funestus'),
    ('Kampala', 0.316, 32.582, 'anopheles', 2024, 'established', 'An. gambiae s.l. / An. funestus'),
    ('Jinja', 0.451, 33.2, 'anopheles', 2023, 'established', 'An. gambiae s.l. / An. funestus'),
    ('Gulu', 2.774, 32.299, 'anopheles', 2023, 'established', 'An. gambiae s.l. / An. funestus'),
    ('Dar es Salaam', -6.792, 39.208, 'anopheles', 2024, 'established', 'An. gambiae s.l. / An. funestus'),
    ('Moshi', -3.355, 37.339, 'anopheles', 2023, 'established', 'An. gambiae s.l. / An. funestus'),
    ('Mbeya', -8.9, 33.46, 'anopheles', 2023, 'established', 'An. gambiae s.l. / An. funestus'),
    ('Ouagadougou', 12.365, -1.533, 'anopheles', 2024, 'established', 'An. gambiae s.l. / An. funestus'),
    ('Bobo-Dioulasso', 11.177, -4.299, 'anopheles', 2023, 'established', 'An. gambiae s.l. / An. funestus'),
    ('Bamako', 12.65, -8.0, 'anopheles', 2024, 'established', 'An. gambiae s.l. / An. funestus'),
    ('Mopti', 14.5, -4.0, 'anopheles', 2022, 'established', 'An. gambiae s.l. / An. funestus'),
    ('Accra', 5.559, -0.197, 'anopheles', 2024, 'established', 'An. gambiae s.l. / An. funestus'),
    ('Kumasi', 6.688, -1.624, 'anopheles', 2023, 'established', 'An. gambiae s.l. / An. funestus'),
    ('Maputo', -25.966, 32.573, 'anopheles', 2024, 'established', 'An. gambiae s.l. / An. funestus'),
    ('Beira', -19.843, 34.838, 'anopheles', 2023, 'established', 'An. gambiae s.l. / An. funestus'),
    ('Nampula', -15.116, 39.267, 'anopheles', 2023, 'established', 'An. gambiae s.l. / An. funestus'),
    ('Yaoundé', 3.866, 11.517, 'anopheles', 2024, 'established', 'An. gambiae s.l. / An. funestus'),
    ('Douala', 4.05, 9.7, 'anopheles', 2023, 'established', 'An. gambiae s.l. / An. funestus'),
    ('Khartoum', 15.552, 32.532, 'anopheles', 2024, 'established', 'An. gambiae s.l. / An. funestus'),
    ('El Fasher', 13.628, 25.349, 'anopheles', 2022, 'established', 'An. gambiae s.l. / An. funestus'),
    ('Jimma', 7.666, 36.834, 'anopheles', 2023, 'established', 'An. gambiae s.l. / An. funestus'),
    ('Hawassa', 7.06, 38.477, 'anopheles', 2023, 'established', 'An. gambiae s.l. / An. funestus'),
    ('Nairobi', -1.292, 36.822, 'anopheles', 2024, 'established', 'An. gambiae s.l. / An. funestus'),
    ('Mombasa', -4.043, 39.668, 'anopheles', 2023, 'established', 'An. gambiae s.l. / An. funestus'),
    ('Kisumu', 0.091, 34.768, 'anopheles', 2023, 'established', 'An. gambiae s.l. / An. funestus'),
    ('Luanda', -8.839, 13.234, 'anopheles', 2024, 'established', 'An. gambiae s.l. / An. funestus'),
    ('Huambo', -12.775, 15.739, 'anopheles', 2023, 'established', 'An. gambiae s.l. / An. funestus'),
    ('Dakar', 14.716, -17.467, 'anopheles', 2024, 'established', 'An. gambiae s.l. / An. funestus'),
    ('Kaolack', 14.163, -15.574, 'anopheles', 2022, 'established', 'An. gambiae s.l. / An. funestus'),
    ('Antananarivo', -18.91, 47.536, 'anopheles', 2023, 'established', 'An. gambiae s.l. / An. funestus'),
    ('Antsiranana', -12.352, 49.301, 'anopheles', 2022, 'established', 'An. gambiae s.l. / An. funestus'),
    ('Juba', 4.859, 31.571, 'anopheles', 2024, 'established', 'An. gambiae s.l. / An. funestus'),
    ('Bangui', 4.361, 18.555, 'anopheles', 2023, 'established', 'An. gambiae s.l. / An. funestus'),
    ('Conakry', 9.538, -13.677, 'anopheles', 2023, 'established', 'An. gambiae s.l. / An. funestus'),
    ('Abidjan', 5.354, -4.008, 'anopheles', 2024, 'established', 'An. gambiae s.l. / An. funestus'),
    ('Monrovia', 6.3, -10.797, 'anopheles', 2023, 'established', 'An. gambiae s.l. / An. funestus'),
    ('Freetown', 8.484, -13.228, 'anopheles', 2023, 'established', 'An. gambiae s.l. / An. funestus'),
    ('Lusaka', -15.417, 28.283, 'anopheles', 2023, 'established', 'An. gambiae s.l. / An. funestus'),
    ('Harare', -17.829, 31.052, 'anopheles', 2023, 'established', 'An. gambiae s.l. / An. funestus'),
    ('Mumbai', 19.076, 72.878, 'anopheles_stephensi', 2024, 'established', 'An. stephensi'),
    ('Delhi', 28.644, 77.216, 'anopheles_stephensi', 2024, 'established', 'An. stephensi'),
    ('Hubli-Dharwad', 15.358, 75.135, 'anopheles_stephensi', 2022, 'established', 'An. stephensi'),
    ('Hyderabad Sindh', 25.396, 68.374, 'anopheles_stephensi', 2022, 'established', 'An. stephensi'),
    ('Quetta', 30.183, 66.998, 'anopheles', 2022, 'established', 'An. culicifacies / An. stephensi'),
    ('Port Moresby', -9.443, 147.18, 'anopheles', 2023, 'established', 'An. farauti / An. punctulatus group'),
    ('PNG Highlands', -6.315, 143.955, 'anopheles', 2022, 'established', 'An. farauti / An. punctulatus group'),
    ('Honiara', -9.428, 160.033, 'anopheles', 2022, 'established', 'An. farauti / An. punctulatus group'),
    ('Port Vila', -17.738, 168.321, 'anopheles', 2022, 'established', 'An. farauti / An. punctulatus group'),
    ('Yangon', 16.871, 96.152, 'anopheles', 2023, 'established', 'An. dirus / An. minimus'),
    ('Mandalay', 21.974, 96.084, 'anopheles', 2022, 'established', 'An. dirus / An. minimus'),
    ('Manaus', -3.1, -60.025, 'anopheles', 2024, 'established', 'An. darlingi'),
    ('Belém', -1.455, -48.503, 'anopheles', 2023, 'established', 'An. darlingi'),
    ('Brasília', -15.793, -47.882, 'aedes_aegypti', 2024, 'established', None),
    ('Rio', -22.908, -43.172, 'aedes_aegypti', 2024, 'established', None),
    ('São Paulo', -23.55, -46.633, 'aedes_aegypti', 2024, 'established', None),
    ('Belo Horizonte', -19.917, -43.934, 'aedes_aegypti', 2024, 'established', None),
    ('Recife', -8.054, -34.881, 'aedes_aegypti', 2024, 'established', None),
    ('Fortaleza', -3.717, -38.543, 'aedes_aegypti', 2024, 'established', None),
    ('Salvador', -12.971, -38.501, 'aedes_aegypti', 2024, 'established', None),
    ('Vitória', -20.319, -40.338, 'aedes_aegypti', 2024, 'established', None),
    ('Curitiba', -25.428, -49.271, 'aedes_aegypti', 2024, 'established', None),
    ('Porto Alegre', -30.033, -51.23, 'aedes_aegypti', 2024, 'established', None),
    ('Manaus', -3.1, -60.025, 'aedes_aegypti', 2024, 'established', None),
    ('Belém', -1.455, -48.503, 'aedes_aegypti', 2024, 'established', None),
    ('Natal', -5.793, -35.209, 'aedes_aegypti', 2024, 'established', None),
    ('Goiânia', -16.687, -49.264, 'aedes_aegypti', 2024, 'established', None),
    ('Buenos Aires', -34.618, -58.381, 'aedes_aegypti', 2024, 'established', None),
    ('Córdoba', -31.417, -64.183, 'aedes_aegypti', 2024, 'established', None),
    ('Mendoza', -32.889, -68.845, 'aedes_aegypti', 2024, 'established', None),
    ('Tucumán', -26.808, -65.217, 'aedes_aegypti', 2024, 'established', None),
    ('Resistencia', -27.45, -58.99, 'aedes_aegypti', 2024, 'established', None),
    ('Posadas', -27.367, -55.896, 'aedes_aegypti', 2024, 'established', None),
    ('Mexico City', 19.432, -99.133, 'aedes_aegypti', 2015, 'detected', None),
    ('Mérida', 20.967, -89.623, 'aedes_aegypti', 2024, 'established', None),
    ('Oaxaca', 17.073, -96.723, 'aedes_aegypti', 2024, 'established', None),
    ('Tuxtla', 16.752, -93.116, 'aedes_aegypti', 2024, 'established', None),
    ('Monterrey', 25.686, -100.316, 'aedes_aegypti', 2024, 'established', None),
    ('Puerto Vallarta', 20.654, -105.227, 'aedes_aegypti', 2023, 'established', None),
    ('Aguascalientes', 21.882, -102.296, 'aedes_aegypti', 2023, 'established', None),
    ('Medellín', 6.244, -75.574, 'aedes_aegypti', 2024, 'established', None),
    ('Cali', 3.871, -76.522, 'aedes_aegypti', 2024, 'established', None),
    ('Barranquilla', 10.964, -74.796, 'aedes_aegypti', 2024, 'established', None),
    ('Bogotá', 4.711, -74.073, 'aedes_aegypti', 2024, 'detected', None),
    ('Bucaramanga', 7.119, -73.123, 'aedes_aegypti', 2023, 'established', None),
    ('Asunción', -25.286, -57.647, 'aedes_aegypti', 2024, 'established', None),
    ('Encarnación', -27.333, -55.867, 'aedes_aegypti', 2023, 'established', None),
    ('Lima', -12.046, -77.043, 'aedes_aegypti', 2024, 'established', None),
    ('Iquitos', -3.743, -73.247, 'aedes_aegypti', 2024, 'established', None),
    ('Guatemala City', 14.641, -90.513, 'aedes_aegypti', 2024, 'established', None),
    ('Tegucigalpa', 14.082, -87.206, 'aedes_aegypti', 2024, 'established', None),
    ('San Pedro Sula', 15.504, -88.025, 'aedes_aegypti', 2024, 'established', None),
    ('Caracas', 10.489, -66.879, 'aedes_aegypti', 2024, 'established', None),
    ('Guayaquil', -2.17, -79.922, 'aedes_aegypti', 2024, 'established', None),
    ('Santa Cruz', -17.783, -63.182, 'aedes_aegypti', 2024, 'established', None),
    ('Managua', 12.136, -86.313, 'aedes_aegypti', 2024, 'established', None),
    ('San Salvador', 13.692, -89.218, 'aedes_aegypti', 2024, 'established', None),
    ('San José CR', 9.93, -84.088, 'aedes_aegypti', 2024, 'established', None),
    ('Port-au-Prince', 18.543, -72.338, 'aedes_aegypti', 2024, 'established', None),
    ('Santo Domingo', 18.486, -69.931, 'aedes_aegypti', 2024, 'established', None),
    ('San Juan PR', 18.466, -66.106, 'aedes_aegypti', 2024, 'established', None),
    ('Havana', 23.136, -82.359, 'aedes_aegypti', 2023, 'established', None),
    ('Kingston', 17.997, -76.79, 'aedes_aegypti', 2023, 'established', None),
    ('Port of Spain', 10.652, -61.519, 'aedes_aegypti', 2023, 'established', None),
    ('Miami', 25.774, -80.193, 'aedes_aegypti', 2024, 'established', None),
    ('Key West', 24.555, -81.781, 'aedes_aegypti', 2024, 'established', None),
    ('Bangkok', 13.754, 100.501, 'aedes_aegypti', 2024, 'established', None),
    ('Kuala Lumpur', 3.14, 101.687, 'aedes_aegypti', 2024, 'established', None),
    ('Ho Chi Minh City', 10.823, 106.63, 'aedes_aegypti', 2024, 'established', None),
    ('Hanoi', 21.028, 105.854, 'aedes_aegypti', 2024, 'established', None),
    ('Manila', 14.599, 120.984, 'aedes_aegypti', 2024, 'established', None),
    ('Jakarta', -6.211, 106.845, 'aedes_aegypti', 2024, 'established', None),
    ('Surabaya', -7.25, 112.75, 'aedes_aegypti', 2024, 'established', None),
    ('Denpasar', -8.65, 115.217, 'aedes_aegypti', 2023, 'established', None),
    ('Singapore', 1.352, 103.82, 'aedes_aegypti', 2024, 'established', None),
    ('Phnom Penh', 11.562, 104.928, 'aedes_aegypti', 2024, 'established', None),
    ('Vientiane', 17.967, 102.6, 'aedes_aegypti', 2023, 'established', None),
    ('Yangon', 16.871, 96.152, 'aedes_aegypti', 2024, 'established', None),
    ('Penang', 5.414, 100.329, 'aedes_aegypti', 2023, 'established', None),
    ('Kuching', 1.54, 110.345, 'aedes_aegypti', 2023, 'established', None),
    ('Dhaka', 23.81, 90.412, 'aedes_aegypti', 2024, 'established', None),
    ('Chittagong', 22.356, 91.783, 'aedes_aegypti', 2024, 'established', None),
    ('Delhi', 28.644, 77.216, 'aedes_aegypti', 2024, 'established', None),
    ('Mumbai', 19.076, 72.878, 'aedes_aegypti', 2024, 'established', None),
    ('Chennai', 13.082, 80.27, 'aedes_aegypti', 2024, 'established', None),
    ('Bengaluru', 12.977, 77.591, 'aedes_aegypti', 2024, 'established', None),
    ('Kolkata', 22.572, 88.363, 'aedes_aegypti', 2024, 'established', None),
    ('Hyderabad', 17.385, 78.487, 'aedes_aegypti', 2024, 'established', None),
    ('Lucknow', 26.85, 80.95, 'aedes_aegypti', 2023, 'established', None),
    ('Colombo', 6.927, 79.861, 'aedes_aegypti', 2024, 'established', None),
    ('Lahore', 31.549, 74.343, 'aedes_aegypti', 2024, 'established', None),
    ('Islamabad', 33.72, 73.043, 'aedes_aegypti', 2024, 'established', None),
    ('Mombasa', -4.043, 39.668, 'aedes_aegypti', 2023, 'established', None),
    ('Djibouti', 11.589, 43.145, 'aedes_aegypti', 2023, 'established', None),
    ('Mogadishu', 2.046, 45.341, 'aedes_aegypti', 2023, 'established', None),
    ('Port Louis', -20.162, 57.499, 'aedes_albopictus', 2009, 'established', None),
    ('St-Denis Réunion', -21.115, 55.536, 'aedes_albopictus', 2005, 'established', None),
    ('Al Hudaydah', 14.798, 42.954, 'aedes_aegypti', 2023, 'established', None),
    ('Aden', 12.779, 45.036, 'aedes_aegypti', 2022, 'established', None),
    ('Cairns', -16.92, 145.77, 'aedes_aegypti', 2023, 'established', None),
    ('Rome', 41.902, 12.496, 'aedes_albopictus', 1997, 'established', None),
    ('Milan', 45.464, 9.19, 'aedes_albopictus', 2024, 'established', None),
    ('Naples', 40.851, 14.268, 'aedes_albopictus', 2024, 'established', None),
    ('Florence', 43.769, 11.255, 'aedes_albopictus', 2024, 'established', None),
    ('Venice', 45.438, 12.335, 'aedes_albopictus', 2024, 'established', None),
    ('Bologna', 44.494, 11.342, 'aedes_albopictus', 2023, 'established', None),
    ('Turin', 45.065, 7.685, 'aedes_albopictus', 2023, 'established', None),
    ('Barcelona', 41.385, 2.173, 'aedes_albopictus', 2004, 'established', None),
    ('Madrid', 40.416, -3.703, 'aedes_albopictus', 2024, 'established', None),
    ('Valencia', 39.47, -0.376, 'aedes_albopictus', 2024, 'established', None),
    ('Marseille', 43.296, 5.37, 'aedes_albopictus', 2024, 'established', None),
    ('Toulouse', 43.605, 1.444, 'aedes_albopictus', 2024, 'established', None),
    ('Bordeaux', 44.837, -0.579, 'aedes_albopictus', 2024, 'established', None),
    ('Nice', 43.71, 7.262, 'aedes_albopictus', 2004, 'established', None),
    ('Thessaloniki', 40.64, 22.944, 'aedes_albopictus', 2024, 'established', None),
    ('Athens', 37.983, 23.727, 'aedes_albopictus', 2024, 'established', None),
    ('Lisbon', 38.717, -9.143, 'aedes_albopictus', 2024, 'established', None),
    ('Porto', 41.158, -8.629, 'aedes_albopictus', 2024, 'established', None),
    ('Zagreb', 45.815, 15.982, 'aedes_albopictus', 2024, 'established', None),
    ('Split', 43.508, 16.44, 'aedes_albopictus', 2023, 'established', None),
    ('Ljubljana', 46.046, 14.506, 'aedes_albopictus', 2024, 'established', None),
    ('Salzburg', 47.809, 13.055, 'aedes_albopictus', 2023, 'established', None),
    ('Budapest', 47.497, 19.04, 'aedes_albopictus', 2024, 'established', None),
    ('Bucharest', 44.439, 26.097, 'aedes_albopictus', 2024, 'established', None),
    ('Sofia', 42.697, 23.322, 'aedes_albopictus', 2024, 'established', None),
    ('Amsterdam', 52.37, 4.895, 'aedes_albopictus', 2024, 'introduced', None),
    ('Munich', 48.137, 11.576, 'aedes_albopictus', 2024, 'established', None),
    ('Valletta', 35.9, 14.51, 'aedes_albopictus', 2023, 'established', None),
    ('Funchal Madeira', 32.65, -16.91, 'aedes_aegypti', 2005, 'established', None),
    ('Las Palmas Gran Canaria', 28.1, -15.41, 'aedes_albopictus', 2024, 'established', None),
    ('Istanbul', 41.015, 28.979, 'aedes_albopictus', 2024, 'established', None),
    ('Trabzon', 41.0, 40.524, 'aedes_albopictus', 2023, 'established', None),
    ('Tokyo', 35.69, 139.69, 'aedes_albopictus', 2014, 'established', None),
    ('Osaka', 34.693, 135.502, 'aedes_albopictus', 2023, 'established', None),
    ('Naha Okinawa', 26.213, 127.681, 'aedes_albopictus', 2023, 'established', None),
    ('Seoul', 37.566, 126.978, 'aedes_albopictus', 2023, 'established', None),
    ('Guangzhou', 23.129, 113.264, 'aedes_albopictus', 2024, 'established', None),
    ('Shenzhen', 22.543, 114.058, 'aedes_albopictus', 2024, 'established', None),
    ('Macau', 22.2, 113.543, 'aedes_albopictus', 2023, 'established', None),
    ('Chengdu', 30.572, 104.066, 'aedes_albopictus', 2023, 'established', None),
    ('Tainan', 22.998, 120.213, 'aedes_albopictus', 2024, 'established', None),
    ('Hong Kong', 22.32, 114.17, 'aedes_albopictus', 2023, 'established', None),
    ('Atlanta', 33.749, -84.388, 'aedes_albopictus', 2024, 'established', None),
    ('Charlotte', 35.227, -80.843, 'aedes_albopictus', 2023, 'established', None),
    ('Nashville', 36.166, -86.781, 'aedes_albopictus', 2023, 'established', None),
    ('Houston', 29.76, -95.37, 'aedes_albopictus', 1985, 'established', None),
    ('South Dakota', 44.367, -100.346, 'culex', 2024, 'established', 'Cx. tarsalis'),
    ('North Dakota', 47.551, -101.002, 'culex', 2024, 'established', 'Cx. tarsalis'),
    ('Nebraska', 41.49, -99.9, 'culex', 2024, 'established', 'Cx. tarsalis'),
    ('Kansas', 39.0, -98.0, 'culex', 2024, 'established', 'Cx. tarsalis'),
    ('Montana', 46.87, -110.36, 'culex', 2024, 'established', 'Cx. tarsalis'),
    ('Wyoming', 43.08, -108.98, 'culex', 2024, 'established', 'Cx. tarsalis'),
    ('Los Angeles', 34.052, -118.243, 'culex', 2024, 'established', 'Cx. quinquefasciatus'),
    ('Dallas', 32.726, -97.321, 'culex', 2024, 'established', 'Cx. quinquefasciatus'),
    ('Chicago', 41.836, -87.684, 'culex_pipiens', 2002, 'established', 'Cx. pipiens'),
    ('New Orleans', 29.951, -90.071, 'culex', 2024, 'established', 'Cx. quinquefasciatus'),
    ('Phoenix', 33.448, -112.074, 'culex', 2024, 'established', 'Cx. quinquefasciatus'),
    ('Milan/Lombardy', 45.464, 9.19, 'culex_pipiens', 2024, 'established', 'Cx. pipiens'),
    ('Venice/Veneto', 45.438, 12.335, 'culex_pipiens', 2024, 'established', 'Cx. pipiens'),
    ('Bologna', 44.494, 11.342, 'culex_pipiens', 2024, 'established', 'Cx. pipiens'),
    ('Rome', 41.902, 12.496, 'culex_pipiens', 2024, 'established', 'Cx. pipiens'),
    ('Naples', 40.851, 14.268, 'culex_pipiens', 2024, 'established', 'Cx. pipiens'),
    ('Turin', 45.065, 7.685, 'culex_pipiens', 2024, 'established', 'Cx. pipiens'),
    ('Thessaloniki', 40.64, 22.944, 'culex_pipiens', 2010, 'established', 'Cx. pipiens'),
    ('Athens', 37.983, 23.727, 'culex_pipiens', 2024, 'established', 'Cx. pipiens'),
    ('Bucharest', 44.439, 26.097, 'culex_pipiens', 1996, 'established', 'Cx. pipiens'),
    ('Belgrade', 44.812, 20.461, 'culex_pipiens', 2024, 'established', 'Cx. pipiens'),
    ('Budapest', 47.497, 19.04, 'culex_pipiens', 2024, 'established', 'Cx. pipiens'),
    ('Volgograd', 48.708, 44.513, 'culex_pipiens', 1999, 'established', 'Cx. pipiens'),
    ('Rostov-on-Don', 47.227, 39.72, 'culex_pipiens', 2023, 'established', 'Cx. pipiens'),
    ('Berlin', 52.52, 13.405, 'culex_pipiens', 2024, 'established', 'Cx. pipiens'),
    ('Amsterdam', 52.37, 4.895, 'culex_pipiens', 2024, 'established', 'Cx. pipiens'),
    ('Jerusalem', 31.771, 35.217, 'culex_pipiens', 2024, 'established', 'Cx. pipiens'),
    ('Tel Aviv', 32.085, 34.781, 'culex_pipiens', 2024, 'established', 'Cx. pipiens'),
    ('Algiers', 36.737, 3.086, 'culex_pipiens', 2023, 'established', 'Cx. pipiens'),
    ('Tunis', 36.819, 10.165, 'culex_pipiens', 2023, 'established', 'Cx. pipiens'),
    ('Tokyo', 35.69, 139.69, 'culex_pipiens', 2024, 'established', 'Cx. pipiens pallens'),
    ('Bangkok', 13.754, 100.501, 'culex', 2024, 'established', 'Cx. quinquefasciatus'),
    ('Delhi', 28.644, 77.216, 'culex', 2024, 'established', 'Cx. quinquefasciatus'),
    ('Kolkata', 22.572, 88.363, 'culex', 2023, 'established', 'Cx. quinquefasciatus'),
    ('Dhaka', 23.81, 90.412, 'culex', 2023, 'established', 'Cx. quinquefasciatus'),
    ('Darwin', -12.462, 130.841, 'culex', 2023, 'established', 'Cx. annulirostris'),
    ('NT interior', -25.86, 130.0, 'culex', 2022, 'established', 'Cx. annulirostris'),
    ('Dar es Salaam', -6.792, 39.208, 'culex', 2023, 'established', 'Cx. quinquefasciatus'),
    ('Lagos', 6.524, 3.379, 'culex', 2023, 'established', 'Cx. quinquefasciatus'),
    ('Kinshasa', -4.322, 15.322, 'culex', 2023, 'established', 'Cx. quinquefasciatus'),
    ('Rio', -22.908, -43.172, 'culex', 2023, 'established', 'Cx. quinquefasciatus'),
    ('Buenos Aires', -34.618, -58.381, 'culex', 2023, 'established', 'Cx. quinquefasciatus'),
    ('Lima', -12.046, -77.043, 'culex', 2022, 'established', 'Cx. quinquefasciatus'),
    ('Djibouti City', 11.589, 43.145, 'anopheles_stephensi', 2012, 'invasive', 'An. stephensi'),
    ('Kebri Dehar', 6.733, 44.267, 'anopheles_stephensi', 2016, 'invasive', 'An. stephensi'),
    ('Arba Minch', 6.033, 37.55, 'anopheles_stephensi', 2022, 'invasive', 'An. stephensi'),
    ('Marsabit', 2.333, 37.983, 'anopheles_stephensi', 2022, 'invasive', 'An. stephensi'),
    ('Lodwar (Turkana)', 3.119, 35.597, 'anopheles_stephensi', 2022, 'invasive', 'An. stephensi'),
    ('Kisumu', -0.092, 34.768, 'anopheles_stephensi', 2022, 'invasive', 'An. stephensi'),
    ('Accra', 5.559, -0.197, 'anopheles_stephensi', 2022, 'invasive', 'An. stephensi'),
    ('Al Hudaydah', 14.798, 42.954, 'anopheles_stephensi', 2021, 'invasive', 'An. stephensi'),
    ('Jaffna', 9.662, 80.026, 'anopheles_stephensi', 2017, 'invasive', 'An. stephensi'),
]

GBIF_TAXA = {   # scientific name -> app species id (colour/legend group)
    'Aedes aegypti': 'aedes_aegypti', 'Aedes albopictus': 'aedes_albopictus',
    'Anopheles gambiae': 'anopheles', 'Anopheles arabiensis': 'anopheles',
    'Anopheles funestus': 'anopheles', 'Anopheles stephensi': 'anopheles_stephensi',
    'Culex quinquefasciatus': 'culex', 'Culex pipiens': 'culex_pipiens',
    'Aedes japonicus': 'aedes_japonicus', 'Aedes vexans': 'aedes_vexans',
    'Mansonia uniformis': 'mansonia_uniformis', 'Culiseta annulata': 'culiseta_annulata',
    'Aedes caspius': 'ochlerotatus_caspius', 'Toxorhynchites brevipalpis': 'toxorhynchites',
}
GBIF_MAX_PER_TAXON = 4000


# ── land mask ────────────────────────────────────────────────────────────────
def _ne(name):
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, name + '.geojson')
    if not os.path.exists(p):
        print(f'  downloading {name} ...')
        urllib.request.urlretrieve(NE + name + '.geojson', p)
    return p

class LandMask:
    def __init__(self):
        from shapely.geometry import shape
        from shapely.strtree import STRtree
        load = lambda n: [shape(f['geometry']) for f in json.load(open(_ne(n)))['features']]
        self.land  = load('ne_10m_land') + load('ne_10m_minor_islands')
        self.lakes = load('ne_10m_lakes')
        self.lt, self.kt = STRtree(self.land), STRtree(self.lakes)

    def _in(self, geoms, tree, p):
        return any(geoms[i].contains(p) for i in tree.query(p))

    def resolve(self, lat, lng):
        """Return (lat, lng, how) with how in {'land','snapped'}, or None to drop."""
        from shapely.geometry import Point
        from shapely.ops import nearest_points
        p = Point(lng, lat)
        if self._in(self.land, self.lt, p) and not self._in(self.lakes, self.kt, p):
            return lat, lng, 'land'
        poly = self.land[self.lt.nearest(p)]
        q = nearest_points(poly.exterior if hasattr(poly, 'exterior') else poly.boundary, p)[0]             if poly.geom_type == 'Polygon' else nearest_points(poly, p)[0]
        km = math.hypot((q.y - lat) * 111.2, (q.x - lng) * 111.2 * math.cos(math.radians(lat)))
        if km > SNAP_KM:
            return None
        # nudge ~150 m inland along the offshore->coast direction so the dot sits on land
        dy, dx = q.y - lat, q.x - lng
        n = math.hypot(dy, dx) or 1.0
        la2, lo2 = q.y + dy / n * 0.0015, q.x + dx / n * 0.0015
        if not self._in(self.land, self.lt, Point(lo2, la2)):
            la2, lo2 = q.y, q.x
        return round(la2, 4), round(lo2, 4), 'snapped'


# ── GBIF (optional, needs network access to api.gbif.org) ─────────────────────
def _get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)

def fetch_gbif():
    recs = []
    for name, sp in GBIF_TAXA.items():
        m = _get('https://api.gbif.org/v1/species/match?' + urllib.parse.urlencode({'name': name}))
        key = m.get('usageKey')
        if not key:
            print(f'  GBIF: no taxon match for {name}'); continue
        got, off, cells = 0, 0, set()
        while got < GBIF_MAX_PER_TAXON:
            q = urllib.parse.urlencode({'taxonKey': key, 'hasCoordinate': 'true',
                 'hasGeospatialIssue': 'false', 'occurrenceStatus': 'PRESENT',
                 'year': '1990,2026', 'limit': 300, 'offset': off})
            page = _get('https://api.gbif.org/v1/occurrence/search?' + q)
            for o in page.get('results', []):
                la, lo = o.get('decimalLatitude'), o.get('decimalLongitude')
                unc = o.get('coordinateUncertaintyInMeters')
                if la is None or lo is None or (unc is not None and unc > 10000):
                    continue
                cell = (round(la, 1), round(lo, 1))
                if cell in cells:
                    continue                      # thin: 1 record / species / 0.1 deg cell
                cells.add(cell)
                place = o.get('locality') or o.get('stateProvince') or o.get('country') or ''
                recs.append((place[:60], la, lo, sp, o.get('year') or 0, 'occurrence',
                             name, 'gbif'))
                got += 1
            if page.get('endOfRecords'):
                break
            off += 300
            time.sleep(0.2)
        print(f'  GBIF {name:28s} {got} thinned records')
    return recs


def build(use_gbif):
    mask = LandMask()
    raw = [r + ('curated',) for r in CURATED]
    if use_gbif:
        raw += fetch_gbif()
    records, seen = [], set()
    stats = {'input': len(raw), 'invalid': 0, 'dropped_at_sea': 0, 'snapped': 0, 'duplicate': 0}
    for place, la, lo, sp, yr, status, taxon, src in raw:
        try:
            la, lo = float(la), float(lo)
        except (TypeError, ValueError):
            stats['invalid'] += 1; continue
        if not (-90 <= la <= 90 and -180 <= lo <= 180) or (la == 0 and lo == 0):
            stats['invalid'] += 1; continue
        res = mask.resolve(la, lo)
        if res is None:
            stats['dropped_at_sea'] += 1; continue
        la, lo, how = res
        if how == 'snapped':
            stats['snapped'] += 1
        k = (sp, round(la, 2), round(lo, 2))
        if k in seen:
            stats['duplicate'] += 1; continue
        seen.add(k)
        records.append({'place': place, 'lat': round(la, 4), 'lng': round(lo, 4), 'sp': sp,
                        'yr': int(yr or 0), 'status': status, 'taxon': taxon, 'src': src})
    stats['output'] = len(records)
    out = {'generated': date.today().isoformat(), 'sources': SOURCES, 'stats': stats,
           'records': records}
    with open(OUT, 'w') as f:
        json.dump(out, f, separators=(',', ':'), ensure_ascii=False)
    print(json.dumps(stats))
    print(f'wrote {OUT} ({os.path.getsize(OUT)//1024} KB)')


if __name__ == '__main__':
    build('--gbif' in sys.argv)
