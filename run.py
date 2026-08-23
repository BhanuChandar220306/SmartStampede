import cv2
import threading
import sqlite3
import json
from datetime import datetime

import ai_engine

from email_alert import send_alert_email
from voice_alert import play_alert_sequence


# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_PATH = (
    "D:/SmartStampede_exp/videos/crowd12.mp4"
)

WINDOW_NAME = (
    "Smart Stampede - AI Crowd Monitoring"
)

GRID_COLS = 4
GRID_ROWS = 2

CAMERA_ID = "CAM07"

# Save historical data every 5 seconds
DATABASE_SAVE_INTERVAL = 5.0

DATABASE_NAME = "crowd_history.db"


# ============================================================
# COLORS - BGR
# ============================================================

GREEN = (
    0,
    220,
    0
)

YELLOW = (
    0,
    220,
    255
)

RED = (
    0,
    0,
    255
)

WHITE = (
    255,
    255,
    255
)

GRAY = (
    150,
    150,
    150
)

BLACK = (
    15,
    15,
    15
)


# ============================================================
# DATABASE
# ============================================================

def initialize_database():

    conn = sqlite3.connect(
        DATABASE_NAME
    )

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS crowd_history (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            timestamp TEXT NOT NULL,

            video_time_seconds REAL,

            camera_id TEXT NOT NULL,

            sector TEXT NOT NULL,

            yolo_count REAL,

            csrnet_count REAL,

            hybrid_count REAL,

            density REAL,

            movement REAL,

            model_disagreement REAL,

            cpi REAL,

            risk_level TEXT,

            danger_sector TEXT,

            highest_density_sector TEXT,

            highest_density REAL,

            highest_sector TEXT,

            highest_cpi REAL,

            fusion_mode TEXT,

            raw_data TEXT
        )
    """)

    conn.commit()
    conn.close()


# ============================================================
# SAFE VALUE CONVERSION
# ============================================================

def safe_float(
    value,
    default=0.0
):

    try:

        if value is None:

            return default

        return float(value)

    except Exception:

        return default


def safe_int(
    value,
    default=0
):

    try:

        if value is None:

            return default

        return int(value)

    except Exception:

        return default


# ============================================================
# MAKE RESULT JSON SAFE
# ============================================================

def make_json_safe(
    value
):

    # --------------------------------------------------------
    # Basic values
    # --------------------------------------------------------

    if value is None:

        return None

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool
        )
    ):

        return value


    # --------------------------------------------------------
    # Dictionary
    # --------------------------------------------------------

    if isinstance(
        value,
        dict
    ):

        output = {}

        for key, item in value.items():

            # Do not save video frame
            if key == "frame":

                continue

            # Do not save huge density maps
            if "density_map" in str(
                key
            ).lower():

                continue

            try:

                output[str(key)] = (
                    make_json_safe(item)
                )

            except Exception:

                output[str(key)] = str(
                    item
                )

        return output


    # --------------------------------------------------------
    # List / Tuple
    # --------------------------------------------------------

    if isinstance(
        value,
        (
            list,
            tuple
        )
    ):

        result = []

        for item in value:

            try:

                result.append(
                    make_json_safe(item)
                )

            except Exception:

                result.append(
                    str(item)
                )

        return result


    # --------------------------------------------------------
    # Numpy / Tensor-like scalar
    # --------------------------------------------------------

    try:

        if hasattr(
            value,
            "item"
        ):

            return value.item()

    except Exception:

        pass


    # --------------------------------------------------------
    # Anything else
    # --------------------------------------------------------

    try:

        return str(
            value
        )

    except Exception:

        return None


# ============================================================
# SAVE HISTORICAL DATA
# ============================================================

def save_historical_data(
    result,
    timestamp,
    video_time_seconds
):

    try:

        # ====================================================
        # GET SECTOR DATA
        # ====================================================

        hybrid_counts = result.get(
            "hybrid_sector_counts",
            {}
        )

        yolo_counts = result.get(
            "yolo_sector_counts",
            {}
        )

        csrnet_counts = result.get(
            "csrnet_sector_counts",
            {}
        )

        sector_cpi = result.get(
            "sector_cpi",
            {}
        )

        sector_risk = result.get(
            "sector_risk",
            {}
        )

        sector_movement = result.get(
            "sector_movement",
            {}
        )

        sector_disagreement = result.get(
            "sector_disagreement",
            {}
        )

        # ----------------------------------------------------
        # Density
        #
        # Different versions of ai_engine may use slightly
        # different names, so check multiple possible names.
        # ----------------------------------------------------

        sector_density = result.get(
            "sector_density_score",
            {}
        )

        if not sector_density:

            sector_density = result.get(
                "sector_density",
                {}
            )

        # ----------------------------------------------------
        # Fusion mode
        # ----------------------------------------------------

        fusion_mode = result.get(
            "fusion_mode",
            {}
        )


        # ====================================================
        # OVERALL INFORMATION
        # ====================================================

        danger_sector = result.get(
            "danger_sector",
            "NONE"
        )

        highest_density_sector = result.get(
            "highest_density_sector",
            "NONE"
        )

        highest_density = safe_float(
            result.get(
                "highest_density",
                0
            )
        )

        highest_sector = result.get(
            "highest_sector",
            "NONE"
        )

        highest_cpi = safe_float(
            result.get(
                "highest_cpi",
                0
            )
        )


        # ====================================================
        # RAW RESULT
        #
        # This preserves additional AI information for future
        # analytics without requiring us to know today what
        # fields the admin may want tomorrow.
        # ====================================================

        safe_result = make_json_safe(
            result
        )

        try:

            raw_data = json.dumps(
                safe_result,
                default=str
            )

        except Exception:

            raw_data = "{}"


        # ====================================================
        # OPEN DATABASE
        # ====================================================

        conn = sqlite3.connect(
            DATABASE_NAME
        )

        cursor = conn.cursor()


        # ====================================================
        # ALL 8 SECTORS
        # ====================================================

        sectors = [
            "A1",
            "A2",
            "A3",
            "A4",
            "B1",
            "B2",
            "B3",
            "B4"
        ]


        for sector in sectors:

            # ------------------------------------------------
            # Get sector values
            # ------------------------------------------------

            yolo = safe_float(
                yolo_counts.get(
                    sector,
                    0
                )
            )

            csrnet = safe_float(
                csrnet_counts.get(
                    sector,
                    0
                )
            )

            hybrid = safe_float(
                hybrid_counts.get(
                    sector,
                    0
                )
            )

            density = safe_float(
                sector_density.get(
                    sector,
                    0
                )
            )

            movement = safe_float(
                sector_movement.get(
                    sector,
                    0
                )
            )

            disagreement = safe_float(
                sector_disagreement.get(
                    sector,
                    0
                )
            )

            cpi = safe_float(
                sector_cpi.get(
                    sector,
                    0
                )
            )

            risk = sector_risk.get(
                sector,
                "NORMAL"
            )

            mode = fusion_mode.get(
                sector,
                "UNKNOWN"
            )


            # ------------------------------------------------
            # Insert
            # ------------------------------------------------

            cursor.execute("""
                INSERT INTO crowd_history (

                    timestamp,

                    video_time_seconds,

                    camera_id,

                    sector,

                    yolo_count,

                    csrnet_count,

                    hybrid_count,

                    density,

                    movement,

                    model_disagreement,

                    cpi,

                    risk_level,

                    danger_sector,

                    highest_density_sector,

                    highest_density,

                    highest_sector,

                    highest_cpi,

                    fusion_mode,

                    raw_data
                )

                VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, (

                timestamp,

                video_time_seconds,

                CAMERA_ID,

                sector,

                yolo,

                csrnet,

                hybrid,

                density,

                movement,

                disagreement,

                cpi,

                risk,

                danger_sector,

                highest_density_sector,

                highest_density,

                highest_sector,

                highest_cpi,

                mode,

                raw_data
            ))


        # ====================================================
        # COMMIT
        # ====================================================

        conn.commit()

        conn.close()


        # ====================================================
        # CONSOLE MESSAGE
        # ====================================================

        print()

        print(
            f"[DATABASE] Saved historical data | "
            f"{timestamp} | "
            f"Video: {video_time_seconds:.1f}s | "
            f"Camera: {CAMERA_ID} | "
            f"Sectors: 8"
        )


    except Exception as e:

        print()

        print(
            "[DATABASE ERROR]:",
            e
        )


# ============================================================
# INITIALIZE DATABASE
# ============================================================

initialize_database()


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(
    VIDEO_PATH
)

if not cap.isOpened():

    print(
        "ERROR: Cannot open video:"
    )

    print(
        VIDEO_PATH
    )

    raise SystemExit


FPS = cap.get(
    cv2.CAP_PROP_FPS
)

if FPS <= 0:

    FPS = 25


WIDTH = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

HEIGHT = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)


# ============================================================
# INFORMATION
# ============================================================

print()

print(
    "=========================================="
)

print(
    " SMART STAMPEDE AI MONITORING"
)

print(
    "=========================================="
)

print(
    "Video      :",
    VIDEO_PATH
)

print(
    "Resolution :",
    WIDTH,
    "x",
    HEIGHT
)

print(
    "FPS        :",
    round(
        FPS,
        2
    )
)

print(
    "Mode       : LIVE"
)

print(
    "Saving     : HISTORICAL DATABASE"
)

print(
    "Database   :",
    DATABASE_NAME
)

print(
    "Timeline   : Every",
    DATABASE_SAVE_INTERVAL,
    "seconds"
)

print(
    "Camera ID  :",
    CAMERA_ID
)

print(
    "Grid       : A1-A4 / B1-B4"
)

print(
    "AI         : YOLO + CSRNet"
)

print(
    "Sector     : CSRNet Density Map + YOLO"
)

print(
    "Risk       : NORMAL / WARNING / ALERT"
)

print(
    "Email      : ENABLED"
)

print(
    "Audio      : SIREN + ENGLISH + TELUGU + HINDI"
)

print(
    "=========================================="
)

print()


# ============================================================
# GRID
# ============================================================

sector_width = (
    WIDTH /
    GRID_COLS
)

sector_height = (
    HEIGHT /
    GRID_ROWS
)


def get_sector_bounds(
    row,
    col
):

    x1 = int(
        col *
        sector_width
    )

    y1 = int(
        row *
        sector_height
    )

    x2 = int(
        (col + 1) *
        sector_width
    )

    y2 = int(
        (row + 1) *
        sector_height
    )

    return (
        x1,
        y1,
        x2,
        y2
    )


def get_sector_name(
    row,
    col
):

    return (
        chr(
            ord("A") +
            row
        )
        +
        str(
            col + 1
        )
    )


# ============================================================
# STATUS COLOR
# ============================================================

def status_color(
    status
):

    if status == "ALERT":

        return RED

    if status == "WARNING":

        return YELLOW

    return GREEN


# ============================================================
# DRAW GRID
# ============================================================

def draw_sector_grid(
    frame,
    result
):

    hybrid_counts = result.get(
        "hybrid_sector_counts",
        {}
    )

    yolo_counts = result.get(
        "yolo_sector_counts",
        {}
    )

    csrnet_counts = result.get(
        "csrnet_sector_counts",
        {}
    )

    sector_cpi = result.get(
        "sector_cpi",
        {}
    )

    sector_risk = result.get(
        "sector_risk",
        {}
    )

    sector_movement = result.get(
        "sector_movement",
        {}
    )


    # ========================================================
    # DRAW EACH SECTOR
    # ========================================================

    for row in range(
        GRID_ROWS
    ):

        for col in range(
            GRID_COLS
        ):

            name = get_sector_name(
                row,
                col
            )

            x1, y1, x2, y2 = (
                get_sector_bounds(
                    row,
                    col
                )
            )

            status = sector_risk.get(
                name,
                "NORMAL"
            )

            color = status_color(
                status
            )


            # ------------------------------------------------
            # Sector border
            # ------------------------------------------------

            if status == "ALERT":

                thickness = 4

            elif status == "WARNING":

                thickness = 3

            else:

                thickness = 1


            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                thickness,
                cv2.LINE_AA
            )


            # ------------------------------------------------
            # Semi-transparent information box
            # ------------------------------------------------

            overlay = frame.copy()

            box_width = 175
            box_height = 78

            cv2.rectangle(
                overlay,

                (
                    x1 + 5,
                    y1 + 5
                ),

                (
                    min(
                        x1 + box_width,
                        x2 - 5
                    ),

                    min(
                        y1 + box_height,
                        y2 - 5
                    )
                ),

                BLACK,
                -1
            )

            cv2.addWeighted(
                overlay,
                0.78,
                frame,
                0.22,
                0,
                frame
            )


            # ------------------------------------------------
            # Sector name
            # ------------------------------------------------

            cv2.putText(
                frame,

                name,

                (
                    x1 + 10,
                    y1 + 22
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.48,

                color,

                2,

                cv2.LINE_AA
            )


            # ------------------------------------------------
            # Hybrid people
            # ------------------------------------------------

            hybrid = float(
                hybrid_counts.get(
                    name,
                    0
                )
            )

            cv2.putText(
                frame,

                f"People: {hybrid:.1f}",

                (
                    x1 + 48,
                    y1 + 22
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.34,

                WHITE,

                1,

                cv2.LINE_AA
            )


            # ------------------------------------------------
            # YOLO / CSRNet
            # ------------------------------------------------

            yolo = int(
                yolo_counts.get(
                    name,
                    0
                )
            )

            csr = float(
                csrnet_counts.get(
                    name,
                    0
                )
            )

            cv2.putText(
                frame,

                f"Y:{yolo}  C:{csr:.1f}",

                (
                    x1 + 10,
                    y1 + 40
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.30,

                WHITE,

                1,

                cv2.LINE_AA
            )


            # ------------------------------------------------
            # CPI
            # ------------------------------------------------

            cpi = float(
                sector_cpi.get(
                    name,
                    0
                )
            )

            cv2.putText(
                frame,

                f"CPI: {cpi:.1f}",

                (
                    x1 + 10,
                    y1 + 57
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.30,

                WHITE,

                1,

                cv2.LINE_AA
            )


            # ------------------------------------------------
            # Status
            # ------------------------------------------------

            cv2.putText(
                frame,

                status,

                (
                    x1 + 85,
                    y1 + 57
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.32,

                color,

                2,

                cv2.LINE_AA
            )


            # ------------------------------------------------
            # Local movement
            # ------------------------------------------------

            movement = float(
                sector_movement.get(
                    name,
                    0
                )
            )

            cv2.putText(
                frame,

                f"Move: {movement:.1f}",

                (
                    x1 + 10,
                    y1 + 72
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.28,

                WHITE,

                1,

                cv2.LINE_AA
            )


# ============================================================
# TOP HUD
# ============================================================

def draw_top_hud(
    frame,
    result
):

    yolo_count = int(
        result.get(
            "yolo_count",
            0
        )
    )

    csrnet_count = float(
        result.get(
            "csrnet_count",
            0
        )
    )

    hybrid_count = float(
        result.get(
            "hybrid_count",
            0
        )
    )

    movement = float(
        result.get(
            "movement_score",
            0
        )
    )

    cpi = float(
        result.get(
            "cpi",
            0
        )
    )

    risk = result.get(
        "risk_level",
        "NORMAL"
    )

    danger_sector = result.get(
        "danger_sector",
        "NONE"
    )


    # ========================================================
    # HUD BACKGROUND
    # ========================================================

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (WIDTH, 62),
        BLACK,
        -1
    )

    cv2.addWeighted(
        overlay,
        0.88,
        frame,
        0.12,
        0,
        frame
    )


    # ========================================================
    # TITLE
    # ========================================================

    cv2.putText(
        frame,

        "AI CROWD MONITORING - CAM 07",

        (10, 18),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.45,

        WHITE,

        1,

        cv2.LINE_AA
    )


    # ========================================================
    # METRICS
    # ========================================================

    cv2.putText(
        frame,

        f"YOLO: {yolo_count}",

        (10, 43),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.34,

        WHITE,

        1,

        cv2.LINE_AA
    )

    cv2.putText(
        frame,

        f"CSRNet: {csrnet_count:.1f}",

        (95, 43),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.34,

        WHITE,

        1,

        cv2.LINE_AA
    )

    cv2.putText(
        frame,

        f"Hybrid: {hybrid_count:.1f}",

        (210, 43),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.34,

        WHITE,

        1,

        cv2.LINE_AA
    )

    cv2.putText(
        frame,

        f"Movement: {movement:.1f}",

        (325, 43),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.34,

        WHITE,

        1,

        cv2.LINE_AA
    )

    cv2.putText(
        frame,

        f"CPI: {cpi:.1f}",

        (465, 43),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.34,

        WHITE,

        1,

        cv2.LINE_AA
    )


    # ========================================================
    # RISK
    # ========================================================

    risk_color = status_color(
        risk
    )

    cv2.putText(
        frame,

        f"RISK: {risk}",

        (WIDTH - 150, 18),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.42,

        risk_color,

        2,

        cv2.LINE_AA
    )


    # ========================================================
    # DANGER SECTOR
    # ========================================================

    danger_color = (
        RED
        if danger_sector != "NONE"
        else GREEN
    )

    cv2.putText(
        frame,

        f"DANGER: {danger_sector}",

        (WIDTH - 150, 43),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.35,

        danger_color,

        2,

        cv2.LINE_AA
    )


# ============================================================
# BOTTOM ALERT
# ============================================================

def draw_bottom_alert(
    frame,
    result
):

    risk = result.get(
        "risk_level",
        "NORMAL"
    )

    danger_sector = result.get(
        "danger_sector",
        "NONE"
    )


    if risk == "ALERT":

        message = (
            f"ALERT - IMMEDIATE RESPONSE "
            f"REQUIRED - SECTOR {danger_sector}"
        )

        color = RED

    elif risk == "WARNING":

        message = (
            f"WARNING - CROWD CONDITION "
            f"REQUIRES ATTENTION - {danger_sector}"
        )

        color = YELLOW

    else:

        message = (
            "CROWD CONDITION NORMAL - "
            "CONTINUOUS AI MONITORING"
        )

        color = GREEN


    # ========================================================
    # BOTTOM BAR
    # ========================================================

    bar_height = 35

    y1 = (
        HEIGHT -
        bar_height
    )

    cv2.rectangle(
        frame,
        (0, y1),
        (WIDTH, HEIGHT),
        color,
        -1
    )


    # ========================================================
    # MESSAGE
    # ========================================================

    text_size = cv2.getTextSize(
        message,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        1
    )[0]

    text_x = max(
        5,
        (
            WIDTH -
            text_size[0]
        ) // 2
    )

    cv2.putText(
        frame,

        message,

        (
            text_x,
            HEIGHT - 12
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.38,

        WHITE,

        1,

        cv2.LINE_AA
    )


# ============================================================
# MAIN LOOP
# ============================================================

frame_number = 0

last_database_save = -DATABASE_SAVE_INTERVAL


# ============================================================
# ALERT STATE
# ============================================================

previous_risk = "NORMAL"


while True:

    # ========================================================
    # READ FRAME
    # ========================================================

    ret, frame = cap.read()

    if not ret:

        break

    frame_number += 1


    # ========================================================
    # VIDEO TIMELINE
    # ========================================================

    video_time_seconds = (
        frame_number /
        FPS
    )


    # ========================================================
    # AI PROCESSING
    # ========================================================

    result = ai_engine.process_frame(
        frame
    )

    if not isinstance(
        result,
        dict
    ):

        print(
            "\nERROR: process_frame() "
            "did not return dictionary."
        )

        break


    # ========================================================
    # SAVE HISTORICAL DATA
    #
    # Save every 5 seconds.
    # ========================================================

    if (
        video_time_seconds -
        last_database_save
        >=
        DATABASE_SAVE_INTERVAL
    ):

        timestamp = (
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )

        save_historical_data(

            result,

            timestamp,

            video_time_seconds
        )

        last_database_save = (
            video_time_seconds
        )


    # ========================================================
    # GET CURRENT RISK
    # ========================================================

    current_risk = result.get(
        "risk_level",
        "NORMAL"
    )


    # ========================================================
    # ALERT DETECTION
    # ========================================================

    # Alert only when system ENTERS ALERT.
    #
    # NORMAL -> ALERT  = ALERT
    # WARNING -> ALERT = ALERT
    # ALERT -> ALERT   = NO NEW ALERT

    if (
        current_risk == "ALERT"
        and
        previous_risk != "ALERT"
    ):

        print()

        print(
            "=========================================="
        )

        print(
            " 🚨 NEW ALERT DETECTED"
        )

        print(
            "=========================================="
        )

        print(
            "Danger Sector:",
            result.get(
                "danger_sector",
                "NONE"
            )
        )

        print(
            "Risk:",
            current_risk
        )

        print(
            "Hybrid People:",
            result.get(
                "hybrid_count",
                0
            )
        )

        print(
            "CPI:",
            result.get(
                "cpi",
                0
            )
        )

        print(
            "=========================================="
        )


        # ====================================================
        # SEND EMAIL
        # ====================================================

        try:

            send_alert_email(
                result
            )

        except Exception as e:

            print(
                "Email alert error:",
                e
            )


        # ====================================================
        # START SIREN + VOICE ALERT
        # ====================================================

        alert_thread = threading.Thread(

            target=play_alert_sequence,

            args=(
                result,
            ),

            daemon=True
        )

        alert_thread.start()


    # ========================================================
    # UPDATE PREVIOUS RISK
    # ========================================================

    previous_risk = current_risk


    # ========================================================
    # GET AI FRAME
    # ========================================================

    display = result.get(
        "frame",
        frame
    )


    # ========================================================
    # DRAW SECTORS
    # ========================================================

    draw_sector_grid(
        display,
        result
    )


    # ========================================================
    # TOP HUD
    # ========================================================

    draw_top_hud(
        display,
        result
    )


    # ========================================================
    # BOTTOM ALERT
    # ========================================================

    draw_bottom_alert(
        display,
        result
    )


    # ========================================================
    # SHOW LIVE
    # ========================================================

    cv2.imshow(
        WINDOW_NAME,
        display
    )


    # ========================================================
    # TERMINAL OUTPUT
    # ========================================================

    yolo_count = result.get(
        "yolo_count",
        0
    )

    csrnet_count = result.get(
        "csrnet_count",
        0
    )

    hybrid_count = result.get(
        "hybrid_count",
        0
    )

    risk = result.get(
        "risk_level",
        "NORMAL"
    )

    danger = result.get(
        "danger_sector",
        "NONE"
    )

    print(
        f"Frame {frame_number:04d} | "
        f"YOLO: {yolo_count:2d} | "
        f"CSRNet: {float(csrnet_count):6.1f} | "
        f"Hybrid: {float(hybrid_count):6.1f} | "
        f"Risk: {risk:7s} | "
        f"Danger: {danger:>2}",
        end="\r"
    )


    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(
        max(
            1,
            int(
                1000 /
                FPS
            )
        )
    ) & 0xFF


    # ========================================================
    # Q = QUIT
    # ========================================================

    if key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()


print()
print()

print(
    f"Finished. Processed "
    f"{frame_number} frames."
)

print(
    "Historical data saved to:",
    DATABASE_NAME
)

print(
    "No output video was saved."
)