"""
roommate_matching.py  —  PHASE 2 STEP 4 (part)
==============================================

City -> State resolution and same-state room matching for the customer
recommendation flow. Works ONLY on DataFrames + plain values (source-agnostic).

City -> State resolution priority:
  1. The data-driven map learned from the tenants dataset
     (utils.build_city_state_map) — adapts automatically to new data.
  2. A static, reusable India city -> state fallback (below) for cities that
     have never appeared among tenants (e.g. a brand-new enquiry city).

Nothing here is hardcoded to the business (no apartment/room/bed/rent values).
The static city->state table is geographic reference data, not business config,
and is easy to extend.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

import pandas as pd

from india_locations import (
    INDIA_CITY_STATE,
    is_india_state,
    location_key,
    resolve_india_state,
)
from utils import normalize_text

# Backward-compatible alias used by recommendation_engine imports.
STATIC_CITY_STATE = INDIA_CITY_STATE


def resolve_state(
    city,
    learned_map: Optional[dict] = None,
    static_map: Optional[dict] = None,
) -> Optional[str]:
    """Resolve a customer's city OR state to a canonical Indian state.

    Priority:
      1. Input is itself an Indian state / alias (e.g. "Tamil Nadu", "TN")
      2. Tenant-learned city → state map
      3. Static India city gazetteer (``static_map`` or INDIA_CITY_STATE)
    Returns None when unresolved.
    """
    key = location_key(city)
    if not key:
        return None

    # State typed directly in the city field.
    india_state = resolve_india_state(city)
    if india_state and is_india_state(city):
        return india_state

    if learned_map:
        learned = {location_key(k): v for k, v in learned_map.items() if location_key(k)}
        if key in learned:
            return learned[key]

    static_map = static_map if static_map is not None else INDIA_CITY_STATE
    if key in static_map:
        return static_map[key]

    # Fallback: city known only via India gazetteer (when custom static_map misses it).
    return india_state


# Tokens that are not real home states (CSV / join artefacts).
_INVALID_STATE_TOKENS = frozenset(
    {"null", "none", "nan", "na", "n/a", "nat", "-", "--", "."}
)


def clean_occupant_state(value) -> Optional[str]:
    """Drop null/NaN/empty junk and canonicalise Indian state names.

    Examples: ``Tamilnadu`` / ``Tamil Nadu `` / ``TN`` → ``Tamil Nadu``.
    Returns None for invalid or empty values.
    """
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text:
        return None
    key = normalize_text(text)
    if not key or key in _INVALID_STATE_TOKENS:
        return None
    canonical = resolve_india_state(text)
    if canonical:
        return canonical
    # Keep unknown but non-junk labels (trimmed, collapsed whitespace).
    return " ".join(text.split())


def clean_occupant_states(values) -> list:
    """Clean a sequence of occupant state values; unique, order preserved."""
    out: list = []
    seen = set()
    if values is None:
        return out
    try:
        if pd.isna(values):
            return out
    except (TypeError, ValueError):
        pass
    if isinstance(values, str):
        values = [s.strip() for s in values.split(";") if s.strip()]
    for v in values:
        cleaned = clean_occupant_state(v)
        if cleaned and cleaned not in seen:
            out.append(cleaned)
            seen.add(cleaned)
    return out


def _states_of(cell) -> list:
    """Normalise an occupant_states cell (list in-memory, or '; ' string from CSV).

    Always strips null/NaN/empty and canonicalises known Indian states.
    """
    if isinstance(cell, list):
        return clean_occupant_states(cell)
    if cell is None:
        return []
    try:
        if pd.isna(cell):
            return []
    except (TypeError, ValueError):
        pass
    return clean_occupant_states(str(cell).split(";"))


def _list_cell(cell) -> list:
    """Parse a list column cell (list in-memory, or '; ' string from CSV)."""
    if isinstance(cell, list):
        return cell
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    return [s.strip() for s in str(cell).split(";") if s.strip()]


def occupants_gender_label(genders) -> str:
    """Room occupants gender mix for display: Male only / Female only / Mixed / None."""
    vals = []
    for g in _list_cell(genders):
        n = normalize_text(g)
        if not n:
            continue
        if n.startswith("m"):
            vals.append("Male")
        elif n.startswith("f"):
            vals.append("Female")
        else:
            vals.append(str(g).strip().title())
    if not vals:
        return "None"
    uniq = set(vals)
    if uniq == {"Male"}:
        return "Male only"
    if uniq == {"Female"}:
        return "Female only"
    if len(uniq) == 1:
        return f"{next(iter(uniq))} only"
    return "Mixed"


def gender_allowed_label(genders) -> str:
    """Gender Allowed from current occupants: Male / Female / Mixed / Unknown.

    Empty occupants -> Unknown (never infer Any). Any only from explicit source.
    """
    vals = []
    for g in _list_cell(genders):
        n = normalize_text(g)
        if not n:
            continue
        if n.startswith("m"):
            vals.append("Male")
        elif n.startswith("f"):
            vals.append("Female")
        else:
            vals.append(str(g).strip().title())
    if not vals:
        return "Unknown"
    uniq = set(vals)
    if uniq == {"Male"}:
        return "Male"
    if uniq == {"Female"}:
        return "Female"
    if len(uniq) == 1:
        return next(iter(uniq))
    return "Mixed"


# Inventory Gender Allowed labels that mean "open to any customer gender".
# Matched case-insensitively; unknown future labels also treated as open.
# "Unknown" = no official policy / no occupants (display), still open for filter.
_OPEN_GENDER_ALLOWED_TOKENS = frozenset(
    {
        "any",
        "unisex",
        "mixed",
        "all",
        "both",
        "open",
        "unknown",
        "na",
        "n/a",
        "none",
    }
)


def classify_gender_allowed(label) -> str:
    """Classify an inventory Gender Allowed value for customer filtering.

    Returns:
      ``male_only``   — exclusive male rooms
      ``female_only`` — exclusive female rooms
      ``open``        — Any / Unisex / Mixed / Unknown / missing / other labels
    """
    n = normalize_text(label)
    if not n or n in _OPEN_GENDER_ALLOWED_TOKENS:
        return "open"
    if n == "female" or n.startswith("female") or n.startswith("fema"):
        return "female_only"
    if n == "male" or (n.startswith("male") and "female" not in n):
        return "male_only"
    if n == "f" or (n.startswith("f") and not n.startswith("m")):
        return "female_only"
    if n == "m" or n.startswith("m"):
        return "male_only"
    return "open"


def _canonical_gender_allowed(label) -> str:
    """Normalize inventory Gender Allowed for assertions / display."""
    policy = classify_gender_allowed(label)
    if policy == "male_only":
        return "Male"
    if policy == "female_only":
        return "Female"
    raw = (
        str(label).strip()
        if label is not None and not (isinstance(label, float) and pd.isna(label))
        else ""
    )
    if not raw or raw.lower() in ("nan", "none"):
        return "Unknown"
    n = normalize_text(raw)
    if n in {"unisex", "mixed"}:
        return "Unisex" if n == "unisex" else "Mixed"
    if n == "unknown":
        return "Unknown"
    # Explicit open-policy tokens only (Any from source, not empty rooms).
    if n in {"any", "all", "both", "open"}:
        return "Any"
    if n in _OPEN_GENDER_ALLOWED_TOKENS:
        return "Unknown"
    return raw


def ensure_gender_allowed_column(rooms: pd.DataFrame) -> pd.DataFrame:
    """Guarantee Gender Allowed exists — from inventory/master only (never occupants)."""
    if rooms is None or not isinstance(rooms, pd.DataFrame) or rooms.empty:
        return rooms
    out = rooms.copy()
    if "gender_allowed" not in out.columns:
        out["gender_allowed"] = "Unknown"
    missing = out["gender_allowed"].isna() | (
        out["gender_allowed"].astype(str).str.strip().isin(["", "nan", "None"])
    )
    out.loc[missing, "gender_allowed"] = "Unknown"
    out["gender_allowed"] = out["gender_allowed"].fillna("Unknown")
    return out


def filter_rooms_for_customer_gender(
    rooms: pd.DataFrame, customer_gender: Optional[str]
) -> pd.DataFrame:
    """Filter recommendable rooms by Customer Gender vs inventory Gender Allowed.

    Must run AFTER available-beds selection and BEFORE ranking.

    Rules:
      Male   → allow Male / Any / Mixed / Unisex / Unknown; reject Female
      Female → allow Female / Any / Mixed / Unisex / Unknown; reject Male
      Any    → no gender filtering
    """
    if rooms is None or not isinstance(rooms, pd.DataFrame) or rooms.empty:
        return rooms

    cg = normalize_text(customer_gender)
    if not cg or cg == "any":
        return ensure_gender_allowed_column(rooms)

    out = ensure_gender_allowed_column(rooms)
    policy = out["gender_allowed"].map(classify_gender_allowed)

    if cg in {"male", "m"} or cg.startswith("male"):
        keep = policy.isin(["male_only", "open"])
        forbidden = "Female"
    elif cg in {"female", "f"} or cg.startswith("female") or cg.startswith("fema"):
        keep = policy.isin(["female_only", "open"])
        forbidden = "Male"
    else:
        return out

    filtered = out.loc[keep].copy()

    # Hard guarantee: no conflicting exclusive gender remains.
    left = filtered["gender_allowed"].map(_canonical_gender_allowed)
    assert not (left == forbidden).any(), (
        f"Gender filter failed: customer={customer_gender!r} still has "
        f"{forbidden} rooms: "
        f"{filtered.loc[left == forbidden, 'room_code'].tolist()}"
    )
    return filtered


def assert_candidates_match_customer_gender(
    candidates: pd.DataFrame, customer_gender: Optional[str]
) -> None:
    """Raise if any candidate conflicts with the selected customer gender."""
    cg = normalize_text(customer_gender)
    if not cg or cg == "any" or candidates is None or candidates.empty:
        return
    if "gender_allowed" not in candidates.columns:
        raise AssertionError("candidates missing gender_allowed after gender filter")
    labels = candidates["gender_allowed"].map(_canonical_gender_allowed)
    if cg in {"male", "m"} or cg.startswith("male"):
        bad = candidates.loc[labels == "Female"]
        assert bad.empty, (
            f"Male customer still has Female rooms: {bad['room_code'].tolist()}"
        )
    elif cg in {"female", "f"} or cg.startswith("female") or cg.startswith("fema"):
        bad = candidates.loc[labels == "Male"]
        assert bad.empty, (
            f"Female customer still has Male rooms: {bad['room_code'].tolist()}"
        )


def occupants_state_summary(states) -> str:
    """Display: 'Tamil Nadu (2), Kerala (1)' or 'None' when empty."""
    vals = [s for s in _states_of(states) if s]
    if not vals:
        return "None"
    counts = Counter(vals)
    return ", ".join(f"{state} ({n})" for state, n in counts.most_common())


def customer_match_label(occupant_states, customer_state: Optional[str]) -> str:
    """Compare customer state vs occupant states (display only).

    Unknown only when there are genuinely no occupants to compare against.
    """
    vals = [s for s in _states_of(occupant_states) if s]
    if not vals:
        return "Unknown"
    if not customer_state:
        return "Unknown"
    target = normalize_text(customer_state)
    same = sum(1 for s in vals if normalize_text(s) == target)
    if same > 0:
        return f"High - {same} occupant(s) from same state"
    return "Low - Different state roommates"


def aggregate_apartment_occupant_field(
    room_inventory: pd.DataFrame, apartment_code, field: str
) -> list:
    """Collect a list field across all rooms in an apartment (for display fallback)."""
    if room_inventory is None or room_inventory.empty or field not in room_inventory.columns:
        return []
    apt = room_inventory[
        room_inventory["apartment_code"].astype(str) == str(apartment_code)
    ]
    out: list = []
    for cell in apt[field]:
        if field == "occupant_states":
            out.extend(_states_of(cell))
        else:
            out.extend(_list_cell(cell))
    return out


def same_state_rooms(room_inventory: pd.DataFrame, state: str) -> pd.DataFrame:
    """Rooms with >=1 available bed whose current occupants include `state`."""
    if state is None:
        return room_inventory.iloc[0:0]
    target = normalize_text(state)
    mask = room_inventory.apply(
        lambda r: r["available_beds"] >= 1
        and target in {normalize_text(s) for s in _states_of(r["occupant_states"])},
        axis=1,
    )
    return room_inventory[mask]


# --------------------------------------------------------------------------- #
# Roommate compatibility score (0-100)
# --------------------------------------------------------------------------- #

def _float_list(cell) -> list:
    out = []
    for v in _list_cell(cell):
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            continue
    return out


def _homogeneity(values: list):
    """Share of the most common category (1.0 = everyone identical). None if empty."""
    vals = [str(v).strip().lower() for v in values if str(v).strip()]
    if not vals:
        return None
    counts = {}
    for v in vals:
        counts[v] = counts.get(v, 0) + 1
    return max(counts.values()) / len(vals)


def _match_ratio(values: list, target):
    """Fraction of occupants equal to `target`. None if not computable."""
    vals = [str(v).strip().lower() for v in values if str(v).strip()]
    if not vals or target is None or not str(target).strip():
        return None
    t = str(target).strip().lower()
    return sum(1 for v in vals if v == t) / len(vals)


def compatibility_scores(
    rooms: pd.DataFrame,
    *,
    customer_state: Optional[str] = None,
    customer_gender: Optional[str] = None,
    customer_occupation: Optional[str] = None,
    customer_age: Optional[float] = None,
) -> pd.Series:
    """Compute a 0-100 roommate-compatibility score for each room.

    Factors (each 0..1) are averaged over ONLY the factors computable for that
    room, so the score strengthens automatically as more tenant data arrives:

      * same_state   : does the room already host a customer-state occupant.
      * gender       : match to the customer's gender if given, else the room's
                       gender homogeneity (a well-defined single-gender room).
      * occupation   : match to the customer's student/working type if given,
                       else the room's occupation homogeneity.
      * age          : closeness to the customer's age if given, else the room's
                       own age cohesion (small spread). Scaled dynamically by the
                       population age spread — never a hardcoded threshold.
      * occupancy    : current occupancy ratio (an established roommate group).

    Everything is dynamic; no hardcoded cities/states/values.
    """
    if rooms.empty:
        return pd.Series([], dtype="float64")

    target_state = normalize_text(customer_state) if customer_state else None

    # Dynamic age scale from the population of occupant ages across these rooms.
    all_ages = []
    if "occupant_ages" in rooms.columns:
        for cell in rooms["occupant_ages"]:
            all_ages.extend(_float_list(cell))
    age_scale = pd.Series(all_ages).std() if len(all_ages) >= 2 else None
    if age_scale is not None and (pd.isna(age_scale) or age_scale <= 0):
        age_scale = None

    def _score_row(r) -> float:
        factors = []

        # same_state (membership; the tier already enforces the hard filter).
        if target_state is not None:
            states = {normalize_text(s) for s in _states_of(r.get("occupant_states"))}
            factors.append(1.0 if target_state in states else 0.0)

        # gender
        genders = _list_cell(r.get("occupant_genders")) if "occupant_genders" in rooms.columns else []
        if genders:
            g = (_match_ratio(genders, customer_gender) if customer_gender
                 else _homogeneity(genders))
            if g is not None:
                factors.append(g)

        # occupation
        occs = _list_cell(r.get("occupant_occupations")) if "occupant_occupations" in rooms.columns else []
        if occs:
            o = (_match_ratio(occs, customer_occupation) if customer_occupation
                 else _homogeneity(occs))
            if o is not None:
                factors.append(o)

        # age
        ages = _float_list(r.get("occupant_ages")) if "occupant_ages" in rooms.columns else []
        if ages and age_scale:
            if customer_age is not None:
                sims = [max(0.0, 1.0 - abs(a - float(customer_age)) / (2 * age_scale)) for a in ages]
                factors.append(sum(sims) / len(sims))
            elif len(ages) >= 2:
                spread = max(ages) - min(ages)
                factors.append(max(0.0, 1.0 - spread / (2 * age_scale)))

        # occupancy ratio (established group)
        cap = r.get("capacity") or 0
        if cap:
            factors.append(min(1.0, float(r.get("occupied_beds", 0)) / float(cap)))

        return round(100 * sum(factors) / len(factors), 1) if factors else 0.0

    return rooms.apply(_score_row, axis=1)
