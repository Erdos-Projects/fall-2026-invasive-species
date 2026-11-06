"""iNaturalist API helpers used by ``build_dataset.py``.

Three jobs:

* ``INat``: a ``requests`` session for https://api.inaturalist.org/v1 that waits between calls (iNaturalist asks
  for about one request per second), retries on rate limits and server errors, and caches taxon records on disk.
* Taxon resolution (``INat.resolve`` and ``species_key_of``): every listed name and every observation is rolled up to
  one *class key*. Subspecies and varieties go to their species, an inactive name follows its replacement, a species
  complex becomes one merged class, and anything above species rank (genus and up) is dropped.
* ``row_from`` / ``fetch_class``: turn the observations of one class into metadata rows, one row per observation,
  using its first photo.
"""
import json
import math
import time
from pathlib import Path

import pandas as pd
import requests

API = "https://api.inaturalist.org/v1"
PLANTAE_ID = 47126
# iNaturalist controlled attribute 12 = "Plant Phenology"
PHENOLOGY = {13: "flowering", 14: "fruiting", 15: "flower budding", 21: "no flowers or fruit"}


def chunks(xs, n):
    """Yield successive lists of at most ``n`` items (the API takes up to 30 taxon ids or 200 observation ids per call)."""
    xs = list(xs)
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


class INat:
    """A requests session with pacing, retries and a persistent taxon cache."""

    def __init__(self, user_agent, pause=1.0, cache_dir=None):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent
        self.pause = pause
        self.n_requests = 0
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.taxa_cache_path = self.cache_dir / "taxa_cache.json" if self.cache_dir else None
        if self.taxa_cache_path and self.taxa_cache_path.exists():
            self.taxa_cache = json.loads(self.taxa_cache_path.read_text())
        else:
            self.taxa_cache = {}

    # ---- plain requests ----------------------------------------------------------------------
    def get(self, path, **params):
        """GET with a pause between calls and backoff on rate limits / server errors."""
        for attempt in range(6):
            try:
                r = self.session.get(f"{API}/{path}", params=params, timeout=60)
            except requests.RequestException as e:
                wait = 5 * (attempt + 1)
                print(f"    network error ({e.__class__.__name__}); retrying in {wait}s"); time.sleep(wait); continue
            if r.status_code == 429 or r.status_code >= 500:
                wait = 15 * (attempt + 1)
                print(f"    HTTP {r.status_code} on {path}; retrying in {wait}s"); time.sleep(wait); continue
            r.raise_for_status()
            self.n_requests += 1
            time.sleep(self.pause)
            return r.json()
        raise RuntimeError(f"giving up on {path} {params}")

    def count(self, **params):
        """Number of observations matching the filters (one call, no records returned)."""
        return self.get("observations", per_page=0, **params)["total_results"]

    def species_counts(self, per_page=500, max_pages=40, **params):
        """All (taxon, count) pairs from /observations/species_counts for the given filters."""
        out = []
        for page in range(1, max_pages + 1):
            res = self.get("observations/species_counts", per_page=per_page, page=page, **params)["results"]
            out += [(r["taxon"], r["count"]) for r in res]
            if len(res) < per_page:
                break
        else:
            print(f"    species_counts: stopped at max_pages={max_pages}; the list may be truncated")
        return out

    def place_name(self, place_id):
        p = self.get(f"places/{place_id}")["results"][0]
        return p.get("display_name") or p.get("name")

    # ---- taxa --------------------------------------------------------------------------------
    def fetch_taxa(self, ids):
        """Full taxon records (with ancestors) for a list of ids, cached on disk."""
        need = [i for i in ids if str(i) not in self.taxa_cache]
        for batch in chunks(need, 30):
            for t in self.get("taxa/" + ",".join(map(str, batch)))["results"]:
                self.taxa_cache[str(t["id"])] = t
            if self.taxa_cache_path:
                self.taxa_cache_path.write_text(json.dumps(self.taxa_cache))
        return {i: self.taxa_cache.get(str(i)) for i in ids}

    def fetch_ranks(self, ids):
        """Lean records (taxon_id, name, rank, rank_level, parent_id) for taxon ids, 30 per call.

        Used for the ancestors of the background species (family / genus); the full records in
        fetch_taxa are ~60 KB each, too heavy to cache for a thousand ancestors."""
        out = []
        for batch in chunks(list(ids), 30):
            for t in self.get("taxa/" + ",".join(map(str, batch)))["results"]:
                out.append(dict(taxon_id=t["id"], name=t["name"], rank=t["rank"], rank_level=t["rank_level"], parent_id=t.get("parent_id")))
        return out

    def resolve(self, tid, depth=0):
        """Return (class_key, note) for a candidate taxon id, or (None, reason).

        inactive taxon -> its single replacement; rank_level < 10 (subspecies, variety) -> the species in
        its ancestry; 10 (species, hybrid) -> itself; 11 (complex) -> itself as a merged class;
        above 11 (genus and up) -> dropped.
        """
        t = self.fetch_taxa([tid])[tid]
        if t is None:
            return None, "taxon id not found"
        if not t.get("is_active", True):
            cur = t.get("current_synonymous_taxon_ids") or []
            if len(cur) == 1 and depth < 3:
                key, note = self.resolve(cur[0], depth + 1)
                return key, f"inactive -> {cur[0]}; {note}"
            return None, f"inactive with {len(cur)} replacements"
        if PLANTAE_ID not in t.get("ancestor_ids", []):
            return None, "not a plant"
        rl = t["rank_level"]
        if rl < 10:
            sp = [a for a in t.get("ancestors", []) if a["rank"] == "species"]
            return (sp[-1]["id"], f"{t['rank']} -> species") if sp else (None, "infraspecific with no species ancestor")
        if rl == 10:
            return t["id"], "species" if t["rank"] == "species" else t["rank"]
        if rl == 11:
            return t["id"], "complex (merged class)"
        return None, f"rank {t['rank']} is above species"


# ---- observations -> rows ------------------------------------------------------------------------
def species_key_of(taxon, complex_keys=(), merge_into_complex=True):
    """Roll an observation's taxon up to its class key (same rules as INat.resolve)."""
    rl = taxon.get("rank_level", 0)
    if rl < 10:
        key = taxon.get("min_species_taxon_id")
    elif rl in (10, 11):
        key = taxon["id"]
    else:
        return None
    if merge_into_complex:
        for a in taxon.get("ancestor_ids", []):
            if a in complex_keys and a != key:
                return a
    return key


def row_from(obs, cls, image_size="medium"):
    """One metadata row for an observation, using its first photo.

    The first block of columns keeps the names used by the earlier versions of the dataset (and by the training
    code); the second block adds what the final dataset needs (class bookkeeping, upload date, phenology,
    observer id, photo size).
    """
    t = obs.get("taxon") or {}
    lng, lat = (obs.get("geojson") or {}).get("coordinates", [None, None])
    photos = obs.get("photos") or []
    p = photos[0]
    ann = [(a.get("controlled_attribute_id"), a.get("controlled_value_id")) for a in obs.get("annotations", [])]
    phen = [PHENOLOGY.get(v) for k, v in ann if k == 12 and v in PHENOLOGY]
    dims = p.get("original_dimensions") or {}
    user = obs.get("user") or {}
    establishment = "introduced" if t.get("introduced") else ("native" if t.get("native") else "")
    return dict(
        label=cls.label, image_path=f"images/{cls.label_dir}/{obs['id']}_{p['id']}.jpg",
        observation_id=obs["id"], observation_url=obs.get("uri"),
        scientific_name=cls.class_name, common_name=cls.common_name, taxon_id=cls.class_key, rank=cls.class_rank,
        establishment_means=establishment,
        latitude=lat, longitude=lng, place_guess=obs.get("place_guess"),
        positional_accuracy_m=obs.get("positional_accuracy"), coordinates_obscured=bool(obs.get("obscured")),
        observed_on=obs.get("observed_on"), quality_grade=obs.get("quality_grade"),
        num_identification_agreements=obs.get("num_identification_agreements"),
        observer=user.get("login"),
        photo_id=p["id"], photo_url=p["url"].replace("square", image_size),
        photo_license=p.get("license_code"), photo_attribution=p.get("attribution"),
        n_photos_in_observation=len(photos),
        # ---- new in the final dataset ----
        class_key=cls.class_key, class_name=cls.class_name, group=cls.group,
        invasive=bool(cls.invasive), native_in_sw=bool(cls.native_in_sw), label04=cls.label04,
        obs_taxon_id=t.get("id"), obs_taxon_name=t.get("name"), obs_rank=t.get("rank"),
        inat_introduced=t.get("introduced"), inat_native=t.get("native"),
        created_at=obs.get("created_at"), time_observed_at=obs.get("time_observed_at"),
        month=int(obs["observed_on"][5:7]) if obs.get("observed_on") else None,
        phenology=";".join(sorted(set(phen))), annotations=json.dumps(ann),
        observer_id=user.get("id"),
        photo_width=dims.get("width"), photo_height=dims.get("height"),
    )


def fetch_class(api, cls, base_filters, per_page=200, max_rows=400, complex_keys=(), merge_into_complex=True,
                image_size="medium"):
    """The newest-uploaded observations of one class, up to max_rows, one row per observation.

    taxon_id=<class_key> (iNaturalist includes descendants, so subspecies and varieties come along),
    hrank as set for the class, order_by created_at desc, page-based paging (max_rows stays far below
    the 10k ceiling). Every observation's taxon is rolled up and must land on the class key; others
    are returned in `skipped`.
    """
    rows, skipped, pages = [], [], 0
    for page in range(1, math.ceil(max_rows / per_page) + 1):
        res = api.get("observations", taxon_id=cls.class_key, hrank=cls.hrank, per_page=per_page, page=page,
                      order_by="created_at", order="desc", **base_filters)["results"]
        pages += 1
        for obs in res:
            if not obs.get("photos"):
                continue
            key = species_key_of(obs.get("taxon") or {}, complex_keys, merge_into_complex)
            if key != cls.class_key:
                skipped.append((obs["id"], (obs.get("taxon") or {}).get("name"), key)); continue
            rows.append(row_from(obs, cls, image_size))
        if len(res) < per_page:
            break
    df = pd.DataFrame(rows)
    if len(df):
        # an upload between two page requests shifts the pages by one row; keep the first copy
        df = df.drop_duplicates("observation_id").head(max_rows).reset_index(drop=True)
    return df, pages, skipped
