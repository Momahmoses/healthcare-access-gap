"""
Synthetic dataset generator for healthcare access gap analysis.
Produces community-level data with facility locations, population density,
road network quality, and travel time estimates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_SEED = 42
N_COMMUNITIES = 3000

STATES = [
    ("Kogi", 7.80, 6.74), ("Niger", 9.93, 6.56), ("Benue", 7.34, 8.78),
    ("Plateau", 9.21, 9.52), ("Nasarawa", 8.50, 8.25), ("Kaduna", 10.52, 7.44),
    ("Bauchi", 10.31, 9.84), ("Taraba", 7.87, 11.37), ("Adamawa", 9.33, 12.40),
    ("Gombe", 10.29, 11.17),
]

FACILITY_TYPES = ["Primary Health Centre", "General Hospital", "Specialist Hospital", "Clinic", "Maternity Ward"]


def generate_community_dataset(n_communities: int = N_COMMUNITIES, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate synthetic community healthcare access dataset."""
    rng = np.random.default_rng(seed)
    records = []

    for comm_id in range(n_communities):
        state, lat, lon = STATES[rng.integers(0, len(STATES))]
        lat_j = lat + rng.normal(0, 0.4)
        lon_j = lon + rng.normal(0, 0.4)

        population = int(max(50, rng.lognormal(7.5, 1.2)))
        is_rural = int(population < 5000)
        road_quality = float(np.clip(rng.beta(2, 3) if is_rural else rng.beta(4, 2), 0.05, 1.0))
        nearest_facility_km = max(0.2, float(rng.exponential(12 if is_rural else 4)))
        facility_type = FACILITY_TYPES[rng.integers(0, len(FACILITY_TYPES))]
        n_facilities_10km = max(0, int(rng.poisson(3 if not is_rural else 0.8)))
        doctors_per_1000 = max(0.0, float(rng.exponential(0.6 if is_rural else 2.5)))
        nurses_per_1000 = max(0.0, float(rng.exponential(1.5 if is_rural else 5.0)))
        beds_per_1000 = max(0.0, float(rng.exponential(0.8 if is_rural else 3.0)))
        health_insurance_pct = float(np.clip(rng.normal(12 if is_rural else 35, 10), 0, 80))

        travel_time_min = (nearest_facility_km / (road_quality * 60 + 10)) * 60
        travel_time_min = float(np.clip(travel_time_min + rng.normal(0, 5), 1, 480))

        coverage_score = (
            min(1.0, n_facilities_10km / 5) * 0.3
            + min(1.0, doctors_per_1000 / 3) * 0.25
            + (1 - min(1.0, travel_time_min / 120)) * 0.25
            + road_quality * 0.1
            + min(1.0, health_insurance_pct / 60) * 0.1
        )
        access_tier = "Well-served" if coverage_score >= 0.6 else "Moderate" if coverage_score >= 0.35 else "Underserved"

        records.append({
            "community_id": comm_id,
            "state": state,
            "latitude": round(lat_j, 6),
            "longitude": round(lon_j, 6),
            "population": population,
            "is_rural": is_rural,
            "road_quality": round(road_quality, 4),
            "nearest_facility_km": round(nearest_facility_km, 2),
            "nearest_facility_type": facility_type,
            "n_facilities_10km": n_facilities_10km,
            "doctors_per_1000": round(doctors_per_1000, 3),
            "nurses_per_1000": round(nurses_per_1000, 3),
            "beds_per_1000": round(beds_per_1000, 3),
            "health_insurance_pct": round(health_insurance_pct, 1),
            "travel_time_min": round(travel_time_min, 1),
            "coverage_score": round(coverage_score, 4),
            "access_tier": access_tier,
        })

    return pd.DataFrame(records)


def save_dataset(output_dir: str | Path = "data/raw") -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = generate_community_dataset()
    path = output_dir / "healthcare_data.csv"
    df.to_csv(path, index=False)
    return path
