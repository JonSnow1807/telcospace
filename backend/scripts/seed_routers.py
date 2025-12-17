#!/usr/bin/env python3
"""
Seed the database with common WiFi router models.

Run with: python scripts/seed_routers.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.session import SessionLocal
from app.crud import router as router_crud
from app.schemas.router import RouterCreate


# Router data with real specifications
ROUTERS = [
    # ==================== Budget Routers ($50-100) ====================
    {
        "model_name": "Archer A6",
        "manufacturer": "TP-Link",
        "frequency_bands": ["2.4GHz", "5GHz"],
        "max_tx_power_dbm": 20,
        "antenna_gain_dbi": 5.0,
        "wifi_standard": "WiFi 5",
        "max_range_meters": 40,
        "coverage_area_sqm": 120,
        "price_usd": 49.99,
        "specs": {"mimo": "2x2", "ports": 4, "beamforming": True}
    },
    {
        "model_name": "RT-AC66U B1",
        "manufacturer": "ASUS",
        "frequency_bands": ["2.4GHz", "5GHz"],
        "max_tx_power_dbm": 20,
        "antenna_gain_dbi": 5.0,
        "wifi_standard": "WiFi 5",
        "max_range_meters": 45,
        "coverage_area_sqm": 140,
        "price_usd": 69.99,
        "specs": {"mimo": "3x3", "ports": 4, "beamforming": True, "aiprotection": True}
    },
    {
        "model_name": "R6700AX",
        "manufacturer": "Netgear",
        "frequency_bands": ["2.4GHz", "5GHz"],
        "max_tx_power_dbm": 21,
        "antenna_gain_dbi": 5.0,
        "wifi_standard": "WiFi 6",
        "max_range_meters": 50,
        "coverage_area_sqm": 150,
        "price_usd": 79.99,
        "specs": {"mimo": "2x2", "ports": 4, "ofdma": True}
    },
    {
        "model_name": "MR7350",
        "manufacturer": "Linksys",
        "frequency_bands": ["2.4GHz", "5GHz"],
        "max_tx_power_dbm": 20,
        "antenna_gain_dbi": 5.0,
        "wifi_standard": "WiFi 6",
        "max_range_meters": 45,
        "coverage_area_sqm": 150,
        "price_usd": 89.99,
        "specs": {"mimo": "2x2", "ports": 4, "mesh_capable": True}
    },

    # ==================== Mid-Range Routers ($100-200) ====================
    {
        "model_name": "RT-AX55",
        "manufacturer": "ASUS",
        "frequency_bands": ["2.4GHz", "5GHz"],
        "max_tx_power_dbm": 22,
        "antenna_gain_dbi": 5.0,
        "wifi_standard": "WiFi 6",
        "max_range_meters": 55,
        "coverage_area_sqm": 180,
        "price_usd": 119.99,
        "specs": {"mimo": "2x2", "ofdma": True, "mu_mimo": True, "beamforming": True}
    },
    {
        "model_name": "Archer AX50",
        "manufacturer": "TP-Link",
        "frequency_bands": ["2.4GHz", "5GHz"],
        "max_tx_power_dbm": 22,
        "antenna_gain_dbi": 5.5,
        "wifi_standard": "WiFi 6",
        "max_range_meters": 60,
        "coverage_area_sqm": 200,
        "price_usd": 129.99,
        "specs": {"mimo": "2x2", "ofdma": True, "ports": 4}
    },
    {
        "model_name": "Nighthawk AX4 (RAX40)",
        "manufacturer": "Netgear",
        "frequency_bands": ["2.4GHz", "5GHz"],
        "max_tx_power_dbm": 23,
        "antenna_gain_dbi": 5.5,
        "wifi_standard": "WiFi 6",
        "max_range_meters": 65,
        "coverage_area_sqm": 220,
        "price_usd": 149.99,
        "specs": {"mimo": "4x4", "ofdma": True, "beamforming": True}
    },
    {
        "model_name": "RT-AX68U",
        "manufacturer": "ASUS",
        "frequency_bands": ["2.4GHz", "5GHz"],
        "max_tx_power_dbm": 23,
        "antenna_gain_dbi": 6.0,
        "wifi_standard": "WiFi 6",
        "max_range_meters": 70,
        "coverage_area_sqm": 250,
        "price_usd": 179.99,
        "specs": {"mimo": "3x3", "ofdma": True, "aiprotection": True, "aimesh": True}
    },
    {
        "model_name": "MX4200",
        "manufacturer": "Linksys",
        "frequency_bands": ["2.4GHz", "5GHz"],
        "max_tx_power_dbm": 23,
        "antenna_gain_dbi": 6.0,
        "wifi_standard": "WiFi 6",
        "max_range_meters": 70,
        "coverage_area_sqm": 270,
        "price_usd": 199.99,
        "specs": {"mimo": "4x4", "mesh_capable": True, "tri_band": False}
    },

    # ==================== High-End Routers ($200-400) ====================
    {
        "model_name": "Nighthawk AX6 (RAX50)",
        "manufacturer": "Netgear",
        "frequency_bands": ["2.4GHz", "5GHz"],
        "max_tx_power_dbm": 24,
        "antenna_gain_dbi": 6.5,
        "wifi_standard": "WiFi 6",
        "max_range_meters": 80,
        "coverage_area_sqm": 300,
        "price_usd": 229.99,
        "specs": {"mimo": "4x4", "ofdma": True, "ports": 5}
    },
    {
        "model_name": "RT-AX86U",
        "manufacturer": "ASUS",
        "frequency_bands": ["2.4GHz", "5GHz"],
        "max_tx_power_dbm": 25,
        "antenna_gain_dbi": 7.0,
        "wifi_standard": "WiFi 6",
        "max_range_meters": 85,
        "coverage_area_sqm": 320,
        "price_usd": 269.99,
        "specs": {"mimo": "4x4", "gaming_mode": True, "2.5g_port": True, "aimesh": True}
    },
    {
        "model_name": "Archer AX90",
        "manufacturer": "TP-Link",
        "frequency_bands": ["2.4GHz", "5GHz", "5GHz"],
        "max_tx_power_dbm": 25,
        "antenna_gain_dbi": 7.0,
        "wifi_standard": "WiFi 6",
        "max_range_meters": 90,
        "coverage_area_sqm": 350,
        "price_usd": 299.99,
        "specs": {"mimo": "4x4", "tri_band": True, "onemesh": True}
    },
    {
        "model_name": "Nighthawk AX12 (RAX120)",
        "manufacturer": "Netgear",
        "frequency_bands": ["2.4GHz", "5GHz"],
        "max_tx_power_dbm": 26,
        "antenna_gain_dbi": 7.5,
        "wifi_standard": "WiFi 6",
        "max_range_meters": 95,
        "coverage_area_sqm": 380,
        "price_usd": 349.99,
        "specs": {"mimo": "8x8", "ofdma": True, "5g_port": True}
    },
    {
        "model_name": "RT-AXE7800",
        "manufacturer": "ASUS",
        "frequency_bands": ["2.4GHz", "5GHz", "6GHz"],
        "max_tx_power_dbm": 25,
        "antenna_gain_dbi": 7.0,
        "wifi_standard": "WiFi 6E",
        "max_range_meters": 90,
        "coverage_area_sqm": 350,
        "price_usd": 399.99,
        "specs": {"mimo": "4x4", "tri_band": True, "160mhz": True, "aimesh": True}
    },

    # ==================== Premium/Enterprise Routers ($400+) ====================
    {
        "model_name": "GT-AXE11000",
        "manufacturer": "ASUS",
        "frequency_bands": ["2.4GHz", "5GHz", "6GHz"],
        "max_tx_power_dbm": 27,
        "antenna_gain_dbi": 8.0,
        "wifi_standard": "WiFi 6E",
        "max_range_meters": 100,
        "coverage_area_sqm": 400,
        "price_usd": 549.99,
        "specs": {"mimo": "4x4", "gaming": True, "10g_port": True, "tri_band": True}
    },
    {
        "model_name": "Nighthawk RAXE500",
        "manufacturer": "Netgear",
        "frequency_bands": ["2.4GHz", "5GHz", "6GHz"],
        "max_tx_power_dbm": 27,
        "antenna_gain_dbi": 8.0,
        "wifi_standard": "WiFi 6E",
        "max_range_meters": 100,
        "coverage_area_sqm": 420,
        "price_usd": 599.99,
        "specs": {"mimo": "8x8", "tri_band": True, "12_stream": True}
    },
    {
        "model_name": "WAX630",
        "manufacturer": "Netgear",
        "frequency_bands": ["2.4GHz", "5GHz", "6GHz"],
        "max_tx_power_dbm": 28,
        "antenna_gain_dbi": 8.5,
        "wifi_standard": "WiFi 6E",
        "max_range_meters": 110,
        "coverage_area_sqm": 450,
        "price_usd": 449.99,
        "specs": {"enterprise": True, "poe": True, "cloud_managed": True}
    },

    # ==================== Enterprise Access Points ====================
    {
        "model_name": "WAP571",
        "manufacturer": "Cisco",
        "frequency_bands": ["2.4GHz", "5GHz"],
        "max_tx_power_dbm": 27,
        "antenna_gain_dbi": 8.0,
        "wifi_standard": "WiFi 6",
        "max_range_meters": 100,
        "coverage_area_sqm": 400,
        "price_usd": 599.99,
        "specs": {"enterprise": True, "poe": True, "captive_portal": True}
    },
    {
        "model_name": "EAP670",
        "manufacturer": "TP-Link",
        "frequency_bands": ["2.4GHz", "5GHz"],
        "max_tx_power_dbm": 26,
        "antenna_gain_dbi": 7.5,
        "wifi_standard": "WiFi 6",
        "max_range_meters": 90,
        "coverage_area_sqm": 380,
        "price_usd": 199.99,
        "specs": {"enterprise": True, "omada_sdn": True, "poe": True}
    },
    {
        "model_name": "U6-Pro",
        "manufacturer": "Ubiquiti",
        "frequency_bands": ["2.4GHz", "5GHz"],
        "max_tx_power_dbm": 26,
        "antenna_gain_dbi": 6.0,
        "wifi_standard": "WiFi 6",
        "max_range_meters": 85,
        "coverage_area_sqm": 350,
        "price_usd": 179.99,
        "specs": {"enterprise": True, "poe": True, "unifi_controller": True}
    },
    {
        "model_name": "U6-Enterprise",
        "manufacturer": "Ubiquiti",
        "frequency_bands": ["2.4GHz", "5GHz", "6GHz"],
        "max_tx_power_dbm": 27,
        "antenna_gain_dbi": 7.0,
        "wifi_standard": "WiFi 6E",
        "max_range_meters": 95,
        "coverage_area_sqm": 400,
        "price_usd": 349.99,
        "specs": {"enterprise": True, "poe": True, "2.5g_port": True}
    },

    # ==================== Mesh Systems (per node) ====================
    {
        "model_name": "Deco XE75 (1-pack)",
        "manufacturer": "TP-Link",
        "frequency_bands": ["2.4GHz", "5GHz", "6GHz"],
        "max_tx_power_dbm": 22,
        "antenna_gain_dbi": 5.0,
        "wifi_standard": "WiFi 6E",
        "max_range_meters": 65,
        "coverage_area_sqm": 230,
        "price_usd": 199.99,
        "specs": {"mesh": True, "ai_driven": True, "tri_band": True}
    },
    {
        "model_name": "ZenWiFi Pro ET12 (1-pack)",
        "manufacturer": "ASUS",
        "frequency_bands": ["2.4GHz", "5GHz", "6GHz"],
        "max_tx_power_dbm": 24,
        "antenna_gain_dbi": 6.0,
        "wifi_standard": "WiFi 6E",
        "max_range_meters": 75,
        "coverage_area_sqm": 280,
        "price_usd": 349.99,
        "specs": {"mesh": True, "aimesh": True, "tri_band": True}
    },
    {
        "model_name": "Orbi RBKE963 (1-pack)",
        "manufacturer": "Netgear",
        "frequency_bands": ["2.4GHz", "5GHz", "6GHz"],
        "max_tx_power_dbm": 25,
        "antenna_gain_dbi": 7.0,
        "wifi_standard": "WiFi 6E",
        "max_range_meters": 85,
        "coverage_area_sqm": 320,
        "price_usd": 499.99,
        "specs": {"mesh": True, "quad_band": True, "10g_port": True}
    },
    {
        "model_name": "Velop MX5300 (1-pack)",
        "manufacturer": "Linksys",
        "frequency_bands": ["2.4GHz", "5GHz", "5GHz"],
        "max_tx_power_dbm": 23,
        "antenna_gain_dbi": 6.0,
        "wifi_standard": "WiFi 6",
        "max_range_meters": 70,
        "coverage_area_sqm": 260,
        "price_usd": 279.99,
        "specs": {"mesh": True, "tri_band": True, "intelligent_mesh": True}
    },
]


def seed_routers():
    """Seed the database with router data."""
    db = SessionLocal()

    try:
        # Check if routers already exist
        existing = router_crud.get_routers(db, limit=1)
        if existing:
            print(f"Database already contains routers. Skipping seed.")
            print(f"Use --force to overwrite existing data.")
            return

        print(f"Seeding {len(ROUTERS)} routers...")

        for router_data in ROUTERS:
            router = RouterCreate(**router_data)
            router_crud.create_router(db, router)
            print(f"  Added: {router_data['manufacturer']} {router_data['model_name']}")

        print(f"\nSuccessfully seeded {len(ROUTERS)} routers!")

    finally:
        db.close()


def clear_and_seed():
    """Clear existing routers and reseed."""
    db = SessionLocal()

    try:
        # Delete all existing routers
        from app.models.router import Router
        db.query(Router).delete()
        db.commit()
        print("Cleared existing routers.")

        # Seed new routers
        for router_data in ROUTERS:
            router = RouterCreate(**router_data)
            router_crud.create_router(db, router)
            print(f"  Added: {router_data['manufacturer']} {router_data['model_name']}")

        print(f"\nSuccessfully seeded {len(ROUTERS)} routers!")

    finally:
        db.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Seed router database")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Clear existing routers and reseed"
    )

    args = parser.parse_args()

    if args.force:
        clear_and_seed()
    else:
        seed_routers()
