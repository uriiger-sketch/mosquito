/* MosquitoNet — species catalogue, geography and solar time-of-day.
 *
 * Classic script (no modules, no build step): top-level declarations are shared
 * with the other scripts on the page through the global lexical scope, exactly
 * as when this code lived inline in index.html. Loaded before detector.js.
 *
 * Contract: fields of SPECIES_DB entries may be ADDED but never renamed or
 * removed — the UI, the server and the federated model all key off them.
 */
'use strict';

// ── Astronomical time-of-day system ──────────────────────────────────────────
// Uses Spencer equation for solar declination + solar noon from longitude.
// Accuracy: ±10 min (sufficient for activity periods).
function solarTimes(latDeg, lngDeg) {
  const now  = new Date();
  const D    = (now - new Date(now.getFullYear(), 0, 0)) / 86400000; // day-of-year
  const B    = (2 * Math.PI / 365) * (D - 81);
  const eqT  = 9.87*Math.sin(2*B) - 7.53*Math.cos(B) - 1.5*Math.sin(B); // minutes
  const decl = 23.45 * Math.sin(B) * Math.PI / 180;                       // radians
  const lat  = latDeg * Math.PI / 180;
  const cosHA = -Math.tan(lat) * Math.tan(decl);
  if (cosHA <= -1) return { sunrise: 0,  sunset: 24 };  // midnight-sun
  if (cosHA >=  1) return { sunrise: 12, sunset: 12 };  // polar-night
  const HA   = Math.acos(cosHA) * 180 / Math.PI / 15;   // hours from solar noon
  const tzOff = -now.getTimezoneOffset() / 60;
  const noon  = 12 - (lngDeg / 15 - tzOff) - eqT / 60;
  return { sunrise: noon - HA, sunset: noon + HA };
}

// Returns 'night' | 'dusk' | 'day' for the current moment at the given location.
// Falls back to a fixed clock-based heuristic when no GPS is available.
function timeOfDayCategory(lat, lng) {
  const now = new Date();
  const h   = now.getHours() + now.getMinutes() / 60;
  const { sunrise, sunset } = (lat != null && lng != null)
    ? solarTimes(lat, lng)
    : { sunrise: 6.5, sunset: 19.5 };
  if (h >= sunrise - 0.5 && h < sunrise + 1.0) return 'dusk';
  if (h >= sunset  - 1.0 && h < sunset  + 0.5) return 'dusk';
  if (h >= sunrise + 1.0 && h < sunset  - 1.0) return 'day';
  return 'night';
}

const SPECIES_DB = [
  // ═══════════════════════════════════════════════════════════════
  // SYMPTOMATIC — Disease vectors. Risk levels per WHO/CDC 2024.
  // Frequencies: Brogdon & McAllister 1998, Rompotis et al. 2015,
  //              Mukundarajan et al. 2017 (Stanford Abuzz dataset),
  //              Fanioudakis et al. 2018, Arthur et al. 2014 (JASA)
  // ═══════════════════════════════════════════════════════════════
  {
    id: 'anopheles',      name: 'Anopheles gambiae',
    common: 'African Malaria Mosquito',
    disease: 'Malaria',   risk: 'CRITICAL',   freq: 406,
    freqRange: [375, 445],  color: '#f43f5e',
    asymptomatic: false,  activity: 'night',  regions: ['Sub-Saharan Africa'],
  },
  {
    id: 'anopheles_stephensi', name: 'Anopheles stephensi',
    common: 'Asian Urban Malaria Mosquito',
    disease: 'Malaria (urban)',  risk: 'CRITICAL',  freq: 388,
    freqRange: [360, 412],  color: '#e11d48',
    asymptomatic: false,  activity: 'night',  regions: ['Middle East / S.Asia / invasive Africa'],
  },
  {
    id: 'aedes_aegypti',  name: 'Aedes aegypti',
    common: 'Yellow Fever Mosquito',
    disease: 'Dengue / Zika / Chikungunya / YF',  risk: 'CRITICAL',  freq: 617,
    freqRange: [580, 655],  color: '#f59e0b',
    asymptomatic: false,  activity: 'day',    regions: ['Tropical / Subtropical'],
  },
  {
    id: 'aedes_albopictus', name: 'Aedes albopictus',
    common: 'Asian Tiger Mosquito',
    disease: 'Dengue / Chikungunya',  risk: 'HIGH',  freq: 668,
    freqRange: [640, 705],  color: '#eab308',
    asymptomatic: false,  activity: 'day',    regions: ['Asia / Invasive Europe & Americas'],
  },
  {
    id: 'culex',          name: 'Culex quinquefasciatus',
    common: 'Southern House Mosquito',
    disease: 'West Nile / Filariasis / JE',  risk: 'HIGH',  freq: 453,
    freqRange: [428, 482],  color: '#3b82f6',
    asymptomatic: false,  activity: 'night',  regions: ['Cosmopolitan tropics'],
  },
  {
    id: 'aedes_japonicus', name: 'Aedes japonicus',
    common: 'Asian Bush Mosquito',
    disease: 'West Nile / Usutu',  risk: 'MEDIUM',  freq: 730,
    freqRange: [705, 760],  color: '#7c3aed',
    asymptomatic: false,  activity: 'day',    regions: ['Invasive Europe / N.America'],
  },
  {
    id: 'aedes_vexans',   name: 'Aedes vexans',
    common: 'Inland Floodwater Mosquito',
    disease: 'Rift Valley Fever / Tularemia',  risk: 'MEDIUM',  freq: 515,
    freqRange: [490, 540],  color: '#10b981',
    asymptomatic: false,  activity: 'dusk',   regions: ['N.America / Europe / Asia'],
  },
  {
    id: 'mansonia_uniformis', name: 'Mansonia uniformis',
    common: 'Rice Field Mosquito',
    disease: 'Brugian filariasis / Rift Valley Fever',  risk: 'MEDIUM',  freq: 570,
    freqRange: [548, 596],  color: '#f97316',
    asymptomatic: false,  activity: 'night',  regions: ['SE Asia / Africa / Pacific'],
  },

  // ═══════════════════════════════════════════════════════════════
  // ASYMPTOMATIC — Non-vector or negligible disease risk.
  // Common nuisance mosquitoes found worldwide.
  // ═══════════════════════════════════════════════════════════════
  {
    id: 'culex_pipiens',  name: 'Culex pipiens',
    common: 'Northern House Mosquito',
    // Primary West Nile virus vector in Europe and North America (also Usutu).
    disease: 'West Nile / Usutu',  risk: 'HIGH',  freq: 594,
    freqRange: [578, 615],  color: '#64748b',
    asymptomatic: false,  activity: 'night',  regions: ['Cosmopolitan temperate'],
  },
  {
    id: 'culiseta_annulata', name: 'Culiseta annulata',
    common: 'Banded Mosquito',
    disease: 'Non-vector (nuisance)',  risk: 'LOW',  freq: 344,
    freqRange: [325, 368],  color: '#94a3b8',
    asymptomatic: true,   activity: 'dusk',   regions: ['Europe / N.Africa / W.Asia'],
  },
  {
    id: 'ochlerotatus_caspius', name: 'Aedes (Ochlerotatus) caspius',
    common: 'Caspian Saltmarsh Mosquito',
    disease: 'Non-vector (mass nuisance)',  risk: 'LOW',  freq: 548,
    freqRange: [525, 570],  color: '#475569',
    asymptomatic: true,   activity: 'dusk',   regions: ['Mediterranean / Middle East / C.Asia'],
  },
  {
    id: 'toxorhynchites', name: 'Toxorhynchites brevipalpis',
    common: 'Elephant Mosquito',
    disease: 'Non-vector (larvae eat other mosquitoes)',  risk: 'NONE',  freq: 290,
    freqRange: [270, 310],  color: '#22d3ee',
    asymptomatic: true,   activity: 'day',    regions: ['Africa / SE Asia / Pacific'],
  },
];

// Genus, derived once from the binomial name ("Aedes (Ochlerotatus) caspius" → Aedes).
SPECIES_DB.forEach(sp => { if (!sp.genus) sp.genus = sp.name.split(' ')[0]; });

// ── Native species by GPS region (WHO/ECDC/CDC distribution 2024) ────────────
function getNativeSpecies(lat, lng) {
  // ══════════════════════════════════════════════════════════════════════════
  // SCIENTIFIC SOURCES:
  // [1] Sinka et al. 2012 — Global distribution of the dominant Anopheles
  //     vectors of human malaria. Parasites & Vectors 5:69
  // [2] Kraemer et al. 2015 — Global compendium of Aedes aegypti & albopictus.
  //     eLife 4:e08347 (global occurrence records 1960-2014)
  // [3] ECDC VectorNet 2024 — Invasive mosquito surveillance in Europe
  //     ecdc.europa.eu/en/disease-vectors/surveillance-and-disease-data/mosquito-maps
  // [4] CDC/USGS 2024 — West Nile Virus & mosquito surveillance (ArboNET)
  // [5] WHO VectorMap 2024 — Country malaria vector reports
  // [6] Iwamura et al. 2020 — Accelerating invasion potential of Ae. japonicus
  //     Nature Communications 11:2014
  // [7] Rezza et al. 2022 — Culex pipiens complex: global distribution review
  // [8] Golding et al. 2015 — Integrating vector control interventions: Mansonia
  //     distribution, Parasites & Vectors 8:99
  // ══════════════════════════════════════════════════════════════════════════

  const s = new Set();
  if (lat === null || lat === undefined || isNaN(lat)) return s;
  const IN = (la0,la1,lo0,lo1) => lat>=la0 && lat<=la1 && lng>=lo0 && lng<=lo1;

  // ══════════════════════════════════════════════════════════════════════════
  // ANOPHELES GAMBIAE complex  [Sinka 2012, WHO VectorMap 2024]
  // Primary range: Sub-Saharan Africa south of Sahara (~17°N) to 35°S.
  // An. superpictus / saccharovi / labranchiae: Mediterranean, Middle East,
  //   Central Asia (separate Anopheles species but same detection frequency).
  // An. sinensis / dirus: South & SE Asia malaria vectors.
  // ══════════════════════════════════════════════════════════════════════════
  // Sub-Saharan Africa (dominant malaria belt)
  if (IN(-35, 17, -18, 52)) s.add('anopheles');
  // E/S Africa extension
  if (IN(-35, 5, 22, 52)) s.add('anopheles');
  // Mediterranean Anopheles (labranchiae/superpictus/saccharovi)
  // Present: Morocco, Algeria, Tunisia, Libya, Egypt, Israel, Lebanon,
  //          Syria, Turkey, Greece, Italy (Sardinia), Spain (historically)
  if (IN(28, 42, -6, 42)) s.add('anopheles');
  // Arabian Peninsula + Iran (An. stephensi already covers, but Anopheles
  // gambiae sensu lato also present in Yemen/Oman/S.Saudi)
  if (IN(12, 28, 36, 62)) s.add('anopheles');
  // South Asia (An. culicifacies, An. stephensi, An. dirus)
  if (IN(5, 32, 65, 95)) s.add('anopheles');
  // SE Asia (An. dirus, An. minimus, An. maculatus, An. balabacensis)
  if (IN(-10, 28, 95, 122)) s.add('anopheles');

  // ══════════════════════════════════════════════════════════════════════════
  // ANOPHELES STEPHENSI  [Sinka 2020, WHO invasive alert 2022]
  // Native: Indian subcontinent, Arabian Peninsula, Iran, Pakistan
  // Invasive since 2012: Horn of Africa (Ethiopia, Djibouti, Sudan, Somalia)
  // Invasive since 2023: Sri Lanka, detected in Ghana, Nigeria
  // ══════════════════════════════════════════════════════════════════════════
  // Native range: Pakistan, India (NW), Iran, Iraq, Gulf states, Yemen, Oman
  if (IN(12, 36, 44, 78)) s.add('anopheles_stephensi');
  // Invasive Horn of Africa
  if (IN(0, 18, 36, 50)) s.add('anopheles_stephensi');

  // ══════════════════════════════════════════════════════════════════════════
  // AEDES AEGYPTI  [Kraemer 2015, PAHO 2024, WHO 2024]
  // Primary range: tropical belt 35°S–35°N worldwide.
  // Northern expansion: SE USA, S.Europe (sporadic), S.Japan.
  // Cannot establish where winter temp <10°C sustained.
  // ══════════════════════════════════════════════════════════════════════════
  // Core tropical belt
  if (lat > -35 && lat < 35 && lng > -110 && lng < 180) s.add('aedes_aegypti');
  if (lat > -35 && lat < 35 && lng > -180 && lng < -30) s.add('aedes_aegypti');
  // Southern USA (Florida, Texas, Gulf Coast, Hawaii) — established [CDC 2024]
  if (IN(24, 37, -100, -65)) s.add('aedes_aegypti');
  // Mediterranean seasonal presence (Madeira, Canary Islands, sporadic)
  if (IN(27, 35, -18,  8)) s.add('aedes_aegypti');
  // E.Mediterranean coast (Turkey, Cyprus, Lebanon) — warming range expansion
  if (IN(34, 38, 26, 37)) s.add('aedes_aegypti');

  // ══════════════════════════════════════════════════════════════════════════
  // AEDES ALBOPICTUS  [Kraemer 2015, ECDC 2024]
  // Native: SE/E Asia. Invasive globally in temperate regions since 1980s.
  // Established: all of Europe south of ~54°N, both Americas, Africa coast,
  //              Middle East, Pacific islands, Oceania.
  // Tolerance: survives winter diapause to ~50°N in urban heat islands.
  // ══════════════════════════════════════════════════════════════════════════
  // Native core: Japan, Korea, China, Taiwan, SE Asia
  if (IN(-10, 50, 100, 148)) s.add('aedes_albopictus');
  // Indian subcontinent + Sri Lanka
  if (IN(5, 35, 65, 100)) s.add('aedes_albopictus');
  // Europe (established south of 54°N, urban heat island push to 58°N)
  if (IN(28, 54, -12, 42)) s.add('aedes_albopictus');
  // Sub-Saharan Africa & E.Africa coast
  if (IN(-35, 20, -18, 55)) s.add('aedes_albopictus');
  // Americas (both coasts, established in 35 US states + all of S.America)
  if (IN(-35, 45, -130, -30)) s.add('aedes_albopictus');
  // Pacific (Hawaii, Pacific Islands, Australia/NZ east coast)
  if (IN(-38, 25, 148, 180)) s.add('aedes_albopictus');

  // ══════════════════════════════════════════════════════════════════════════
  // CULEX QUINQUEFASCIATUS (Southern House Mosquito)  [WHO 2024, Rezza 2022]
  // Range: tropical + warm subtropical worldwide, roughly 35°S–35°N.
  // Dominant WNV bridge vector in Americas; filariasis vector in Asia/Africa.
  // Does NOT establish in cold-winter temperate climates (≠ Cx. pipiens).
  // ══════════════════════════════════════════════════════════════════════════
  if (lat > -35 && lat < 36) s.add('culex');
  // S.USA (Florida, Texas, California) — established
  if (IN(24, 37, -125, -65)) s.add('culex');

  // ══════════════════════════════════════════════════════════════════════════
  // AEDES JAPONICUS  [Iwamura 2020, ECDC VectorNet 2024]
  // Native: Japan, Korea, N.China (temperate E.Asia)
  // Invasive NE America: established since 1998, range expanding N/W
  // Invasive C.Europe: established in Germany/Austria/Switzerland since ~2000;
  //   spreading to Belgium, Netherlands, Czech Republic, N.Italy, E.France.
  //   As of 2024: confirmed in 12 European countries.
  // NOT established in Mediterranean, Middle East, UK (only sporadic finds).
  // ══════════════════════════════════════════════════════════════════════════
  // Native E.Asia
  if (IN(30, 46, 120, 145)) s.add('aedes_japonicus');
  // Invasive C.Europe core (confirmed breeding populations)
  if (IN(46, 55,  6,  17)) s.add('aedes_japonicus'); // D/A/CH core
  if (IN(49, 55, 10,  24)) s.add('aedes_japonicus'); // Germany, Czech, Poland
  if (IN(44, 50,  5,  16)) s.add('aedes_japonicus'); // N.Italy, S.Germany, France
  if (IN(50, 56,  2,  10)) s.add('aedes_japonicus'); // Belgium, Netherlands
  // Invasive NE North America
  if (IN(35, 55, -90, -60)) s.add('aedes_japonicus');

  // ══════════════════════════════════════════════════════════════════════════
  // AEDES VEXANS (Inland Floodwater Mosquito)  [multiple CDC/ECDC reports]
  // Range: nearly all N.hemisphere above 20°N; sparse presence in S.hemisphere.
  // Opportunistic floodwater breeder — population spikes after rain/floods.
  // ══════════════════════════════════════════════════════════════════════════
  if (lat > 20 && lat < 72) s.add('aedes_vexans');
  // S.hemisphere presence (Argentina, S.Africa) — limited
  if (IN(-38, -20, -70, -30)) s.add('aedes_vexans');

  // ══════════════════════════════════════════════════════════════════════════
  // MANSONIA UNIFORMIS  [Golding 2015, WHO Global Vector Hub 2024]
  // Strictly tied to aquatic macrophytes (Pistia, Eichhornia).
  // Range: SE Asia (indochina, insular SE Asia, India), tropical Africa,
  //        NE Australia. NOT in arid/desert regions.
  // ══════════════════════════════════════════════════════════════════════════
  // SE Asia mainland + insular SE Asia
  if (IN(-8, 28, 95, 130)) s.add('mansonia_uniformis');
  // Indian subcontinent (coastal + river floodplains)
  if (IN(5, 25, 70, 95)) s.add('mansonia_uniformis');
  // Tropical Africa (W/C/E Africa, NOT Sahara/arid zones)
  if (IN(-12, 15, -18, 42)) s.add('mansonia_uniformis');
  // NE Australia
  if (IN(-28, -10, 130, 155)) s.add('mansonia_uniformis');

  // ══════════════════════════════════════════════════════════════════════════
  // CULEX PIPIENS (Northern House Mosquito)  [Rezza 2022, ECDC 2024]
  // Cosmopolitan temperate; dominant WNV vector in Europe/N.Africa/W.Asia.
  // Range: practically everywhere above 25°N and below 25°S; also in tropical
  //        zones as Cx. pipiens molestus (underground biotype).
  // Note: overlaps with Cx. quinquefasciatus in subtropical transition zone.
  // ══════════════════════════════════════════════════════════════════════════
  if (lat > 12) s.add('culex_pipiens');   // N.hemisphere (incl. tropical)
  if (lat < -10) s.add('culex_pipiens');  // S.hemisphere
  // Deep tropics: Cx. pipiens molestus biotype present in most cities worldwide
  if (lat >= -12 && lat <= 12) s.add('culex_pipiens');

  // ══════════════════════════════════════════════════════════════════════════
  // CULISETA ANNULATA  [ECDC Species Distribution 2024]
  // Strictly N.Europe + N.Africa winter-tolerant species.
  // Range: British Isles, Scandinavia (to ~65°N), France, Germany, Poland,
  //        Iberian Peninsula, N.Italy, N.Turkey, Caucasus, N.Morocco/Algeria.
  // NOT in Scandinavia above treeline; NOT in Mediterranean tropics.
  // ══════════════════════════════════════════════════════════════════════════
  if (IN(48, 66, -10, 32)) s.add('culiseta_annulata'); // N.Europe core
  if (IN(40, 52, -10,  6)) s.add('culiseta_annulata'); // British Isles + NW France/Iberia
  if (IN(35, 45, -10, 15)) s.add('culiseta_annulata'); // S.Iberia, N.Morocco, N.Tunisia

  // ══════════════════════════════════════════════════════════════════════════
  // AEDES (OCHLEROTATUS) CASPIUS  [ECDC VectorNet 2024, Schaffner 2012]
  // Saltmarsh/brackish coastal breeder.
  // Range: Mediterranean basin (all coasts + islands), Caspian Sea basin,
  //        Black Sea coasts, Middle East (Levant, Israel, Jordan, Egypt delta),
  //        Canary Islands, N.Africa coast, Iran coast, Pakistan coast.
  // NOT found far inland or in N.Europe above ~50°N.
  // ══════════════════════════════════════════════════════════════════════════
  if (IN(28, 48,  -8, 55)) s.add('ochlerotatus_caspius'); // Med + Black Sea + Caspian
  if (IN(28, 42, -18,  0)) s.add('ochlerotatus_caspius'); // Canary Islands + N.Atlantic
  if (IN(20, 32,  28, 62)) s.add('ochlerotatus_caspius'); // Nile delta + Red Sea + Gulf

  // ══════════════════════════════════════════════════════════════════════════
  // TOXORHYNCHITES BREVIPALPIS (Elephant Mosquito)
  // Non-biting predatory species. Humid tropics only.
  // Range: sub-Saharan Africa, SE Asia, Pacific islands, N.Australia.
  // ══════════════════════════════════════════════════════════════════════════
  if (IN(-20, 25, 25, 180)) s.add('toxorhynchites');
  if (IN(-20, 15, -18, 25)) s.add('toxorhynchites'); // W/C Africa

  return s;
}


// ── getLocalDisease(speciesId, lat, lng) ─────────────────────────────────────
// Returns the disease(s) actually transmitted by this species IN THIS REGION.
// A species may carry a pathogen globally but only transmit it where the
// pathogen is also present. Sources: WHO Disease Maps 2024, CDC ArboNET,
// ECDC Threat Reports, ProMED Disease Watch.
function getLocalDisease(speciesId, lat, lng) {
  if (lat === null || lat === undefined) return null; // use default from SPECIES_DB
  const IN = (la0,la1,lo0,lo1) => lat>=la0&&lat<=la1&&lng>=lo0&&lng<=lo1;

  switch (speciesId) {

    case 'anopheles':
    case 'anopheles_stephensi':
      // Malaria is actively transmitted wherever these mosquitoes live
      // (Mediterranean species: historic risk, now very low in EU/Israel)
      if (IN(28,42,-6,42)) return 'Malaria (low risk)'; // Med/Middle East: low now
      return 'Malaria';

    case 'aedes_aegypti':
      // Dengue/Zika/Chikungunya only where these viruses circulate.
      // In Med/Middle East the mosquito may be present but no active circulation.
      if (IN(-35, 5,-82,-33)) return 'Dengue · Zika · Chikungunya';    // S.America
      if (IN( -5,25,-100,-60)) return 'Dengue · Chikungunya';           // C.America
      if (IN(-15, 5,-18, 12)) return 'Yellow Fever · Dengue';           // W.Africa
      if (IN(-10,15, 12, 52)) return 'Yellow Fever · Dengue';           // E/C Africa
      if (IN(  5,28, 65,122)) return 'Dengue · Chikungunya · Zika';     // S/SE Asia
      if (IN( 20,32,100,125)) return 'Dengue · Chikungunya';            // E.Asia/Taiwan/Japan
      // Americas Caribbean/Gulf
      if (IN( 10,35,-100,-60)) return 'Dengue · Chikungunya · Zika';
      // Mediterranean/Middle East: vector present, no endemic circulation
      if (IN( 28,42, -6, 42)) return 'Present (no endemic outbreak)';
      return 'Present (no active outbreak)';

    case 'aedes_albopictus':
      if (IN(  5,28, 65,122)) return 'Dengue · Chikungunya';           // SE/E Asia endemic
      if (IN(-10,15, 12, 52)) return 'Chikungunya · Dengue';           // E.Africa
      if (IN(-35, 5,-82,-33)) return 'Dengue · Chikungunya';           // S.America
      if (IN( 10,35,-100,-60)) return 'Dengue · Chikungunya';          // C.America/Caribbean
      if (IN( 28,50, -8, 42)) return 'Chikungunya risk · WNV';         // S.Europe/Med: limited
      return 'Potential vector · low current risk';

    case 'culex':
      // Culex quinquefasciatus vectors differ by region
      if (IN(  5,32, 65, 95)) return 'Japanese Encephalitis · Filariasis'; // S.Asia JE belt
      if (IN(-10,25, 95,125)) return 'Japanese Encephalitis · Filariasis'; // SE Asia JE belt
      if (IN(-35, 5, -18, 52)) return 'Filariasis';                    // Sub-Saharan Africa
      if (IN( 25,52,-130,-60)) return 'West Nile Virus';               // USA/Canada WNV
      if (IN( 15,55, -15, 65)) return 'West Nile Virus';               // Europe/Med/M.East
      return 'West Nile Virus';

    case 'aedes_japonicus':
      // Competent WNV/Usutu vector but primary circulation via Culex
      if (IN( 30,46,120,145)) return 'West Nile · SFTS virus';         // native Japan/Korea
      return 'West Nile · Usutu (low risk)';

    case 'aedes_vexans':
      // Secondary vector, actual disease transmission is very regional
      if (IN(-15,25,-18,52)) return 'Rift Valley Fever';               // Africa
      if (IN( 15,55,-15,65)) return 'West Nile (bridging vector)';     // Europe/M.East
      if (IN( 20,60,-110,-60)) return 'West Nile (bridging vector)';   // N.America
      return 'Rift Valley Fever · secondary WNV';

    case 'mansonia_uniformis':
      if (IN(-8,28,95,130)) return 'Brugian Filariasis';               // SE Asia primary
      if (IN( 5,25,65,95))  return 'Brugian Filariasis';               // India
      return 'Filariasis';

    default:
      return null; // use SPECIES_DB default
  }
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { SPECIES_DB, getNativeSpecies, getLocalDisease, solarTimes, timeOfDayCategory };
}
