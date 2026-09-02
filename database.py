import sqlite3
from datetime import datetime

DATABASE_NAME = "crowd_history.db"


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection():
    return sqlite3.connect(DATABASE_NAME)


# ============================================================
# CREATE TABLE
# ============================================================

def initialize_database():

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS crowd_history (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            timestamp TEXT NOT NULL,

            camera_id TEXT NOT NULL,

            sector TEXT NOT NULL,

            yolo_count REAL,

            csrnet_count REAL,

            hybrid_count REAL,

            density_score REAL,

            movement REAL,

            model_disagreement REAL,

            cpi REAL,

            risk_level TEXT,

            danger_sector TEXT,

            highest_density_sector TEXT,

            highest_density REAL,

            highest_sector TEXT,

            highest_cpi REAL,

            fusion_mode TEXT
        )
    """)

    conn.commit()
    conn.close()


# ============================================================
# SAVE ONE SECTOR RECORD
# ============================================================

def save_sector_data(
    timestamp,
    camera_id,
    sector,
    yolo_count,
    csrnet_count,
    hybrid_count,
    density_score,
    movement,
    model_disagreement,
    cpi,
    risk_level,
    danger_sector,
    highest_density_sector,
    highest_density,
    highest_sector,
    highest_cpi,
    fusion_mode
):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO crowd_history (

            timestamp,
            camera_id,
            sector,

            yolo_count,
            csrnet_count,
            hybrid_count,

            density_score,
            movement,
            model_disagreement,

            cpi,
            risk_level,

            danger_sector,

            highest_density_sector,
            highest_density,

            highest_sector,
            highest_cpi,

            fusion_mode
        )

        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (

        timestamp,
        camera_id,
        sector,

        yolo_count,
        csrnet_count,
        hybrid_count,

        density_score,
        movement,
        model_disagreement,

        cpi,
        risk_level,

        danger_sector,

        highest_density_sector,
        highest_density,

        highest_sector,
        highest_cpi,

        fusion_mode
    ))

    conn.commit()
    conn.close()