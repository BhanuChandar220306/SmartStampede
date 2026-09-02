import cv2
import numpy as np
import torch
from torchvision import transforms
from ultralytics import YOLO

from csrnet import CSRNet


# ============================================================
# CONFIGURATION
# ============================================================

# Better YOLO model for dense crowds.
# You already have this file.
YOLO_MODEL_PATH = "yolov8s.pt"

# Larger image improves small-person detection,
# but is slower on CPU.
YOLO_IMGSZ = 960

# Lower confidence helps detect partially visible people.
YOLO_CONF = 0.25

# 4 x 2 dynamic sectors:
#
# A1 | A2 | A3 | A4
# -----------------
# B1 | B2 | B3 | B4
#
GRID_COLS = 4
GRID_ROWS = 2

SECTORS = [
    "A1", "A2", "A3", "A4",
    "B1", "B2", "B3", "B4"
]

CSRNET_INTERVAL = 5


# ============================================================
# RISK THRESHOLDS
# ============================================================
#
# IMPORTANT:
# These are NOT "sector capacity".
#
# They are starting thresholds for crowd-density risk.
# They should be calibrated using your actual CCTV videos.
#
# Density is represented as:
#
#       people / 100,000 pixels
#
# This avoids the old artificial:
#
#       SECTOR_CAPACITY = 25
#
# ============================================================

DENSITY_WARNING = 24.0
DENSITY_ALERT = 48.0

# Minimum hybrid crowd estimate before movement
# can independently trigger an alert.
MIN_PEOPLE_FOR_MOVEMENT = 4.0

# Risk score thresholds
WARNING_SCORE = 45.0
ALERT_SCORE = 70.0


# ============================================================
# LOAD YOLO
# ============================================================

print("Loading YOLO...")

yolo_model = YOLO(
    YOLO_MODEL_PATH
)

# Keep compatibility with the environment
# where automatic fusion previously caused problems.
try:
    yolo_model.model.fuse = (
        lambda verbose=False: yolo_model.model
    )
except Exception:
    pass

print("YOLO loaded successfully")


# ============================================================
# LOAD CSRNET
# ============================================================

print("Loading CSRNet...")

csrnet_model = CSRNet()

checkpoint = torch.load(
    "weights.pth",
    map_location="cpu"
)

# ------------------------------------------------------------
# Extract state_dict if checkpoint is wrapped
# ------------------------------------------------------------

if isinstance(checkpoint, dict):

    if "state_dict" in checkpoint:

        checkpoint = checkpoint["state_dict"]

    elif "model_state_dict" in checkpoint:

        checkpoint = checkpoint["model_state_dict"]


# ------------------------------------------------------------
# Remove "module." prefix if checkpoint was saved
# using DataParallel.
# ------------------------------------------------------------

clean_checkpoint = {}

for key, value in checkpoint.items():

    if key.startswith("module."):

        key = key[7:]

    clean_checkpoint[key] = value


missing_keys, unexpected_keys = (
    csrnet_model.load_state_dict(
        clean_checkpoint,
        strict=False
    )
)

csrnet_model.eval()

print("CSRNet loaded successfully")

if missing_keys:

    print(
        "WARNING: CSRNet missing keys:",
        len(missing_keys)
    )

if unexpected_keys:

    print(
        "WARNING: CSRNet unexpected keys:",
        len(unexpected_keys)
    )


# ============================================================
# CSRNET TRANSFORM
# ============================================================

csrnet_transform = transforms.Compose([

    transforms.ToPILImage(),

    transforms.Resize(
        (512, 512)
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# STATE
# ============================================================

csrnet_counter = 0

last_csrnet_count = 0.0

last_csrnet_density_map = None

previous_gray = None

previous_sector_movement = {
    sector: 0.0
    for sector in SECTORS
}

# Temporal smoothing prevents:
#
# NORMAL -> ALERT -> NORMAL -> ALERT
#
# on every frame.
sector_score_ema = {
    sector: 0.0
    for sector in SECTORS
}


# ============================================================
# BASIC HELPERS
# ============================================================

def create_sector_dict(value=0):

    return {
        sector: value
        for sector in SECTORS
    }


def create_sector_boxes():

    return {
        sector: []
        for sector in SECTORS
    }


# ============================================================
# GET SECTOR
# ============================================================

def get_sector(
    x,
    y,
    width,
    height
):

    sector_width = (
        width / GRID_COLS
    )

    sector_height = (
        height / GRID_ROWS
    )

    col = int(
        x / sector_width
    )

    row = int(
        y / sector_height
    )

    col = max(
        0,
        min(
            GRID_COLS - 1,
            col
        )
    )

    row = max(
        0,
        min(
            GRID_ROWS - 1,
            row
        )
    )

    return (
        chr(
            ord("A") + row
        )
        +
        str(col + 1)
    )


# ============================================================
# SECTOR BOUNDS
# ============================================================

def get_sector_bounds(
    sector,
    width,
    height
):

    row = (
        ord(sector[0])
        -
        ord("A")
    )

    col = (
        int(sector[1])
        -
        1
    )

    x1 = int(
        col *
        width /
        GRID_COLS
    )

    x2 = int(
        (col + 1) *
        width /
        GRID_COLS
    )

    y1 = int(
        row *
        height /
        GRID_ROWS
    )

    y2 = int(
        (row + 1) *
        height /
        GRID_ROWS
    )

    return (
        x1,
        y1,
        x2,
        y2
    )


# ============================================================
# CSRNET DENSITY MAP
# ============================================================
#
# IMPORTANT:
#
# CSRNet does NOT just return a single number here.
#
# It returns:
#
#       density_map
#
# Then we divide that density map into:
#
# A1 A2 A3 A4
# B1 B2 B3 B4
#
# This is the important correction.
# ============================================================

def calculate_csrnet_density_map(
    frame
):

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    tensor = csrnet_transform(
        rgb
    )

    tensor = tensor.unsqueeze(0)

    with torch.no_grad():

        density_map = csrnet_model(
            tensor
        )

    density_map = (
        density_map
        .squeeze()
        .cpu()
        .numpy()
    )

    # Density should not be negative.
    density_map = np.maximum(
        density_map,
        0
    )

    return density_map.astype(
        np.float32
    )


# ============================================================
# CSRNET SECTOR COUNTS
# ============================================================

def density_map_to_sector_counts(
    density_map
):

    sector_counts = (
        create_sector_dict(
            0.0
        )
    )

    if density_map is None:

        return sector_counts

    map_height, map_width = (
        density_map.shape[:2]
    )

    sector_width = (
        map_width /
        GRID_COLS
    )

    sector_height = (
        map_height /
        GRID_ROWS
    )

    for row in range(
        GRID_ROWS
    ):

        for col in range(
            GRID_COLS
        ):

            x1 = int(
                col *
                sector_width
            )

            x2 = int(
                (col + 1) *
                sector_width
            )

            y1 = int(
                row *
                sector_height
            )

            y2 = int(
                (row + 1) *
                sector_height
            )

            sector = (
                chr(
                    ord("A") + row
                )
                +
                str(col + 1)
            )

            region = (
                density_map[
                    y1:y2,
                    x1:x2
                ]
            )

            count = float(
                region.sum()
            )

            sector_counts[
                sector
            ] = max(
                count,
                0.0
            )

    return sector_counts


# ============================================================
# CSRNET TOTAL
# ============================================================

def calculate_csrnet_count(
    frame
):

    density_map = (
        calculate_csrnet_density_map(
            frame
        )
    )

    count = float(
        density_map.sum()
    )

    return (
        max(count, 0.0),
        density_map
    )


# ============================================================
# YOLO + CSRNET FUSION
# ============================================================
#
# YOLO:
# Good at individual people.
#
# CSRNet:
# Better for heavily occluded crowds.
#
# We do NOT let YOLO alone determine the sector.
#
# We fuse them sector-by-sector.
# ============================================================

def fuse_sector_counts(
    yolo_sector_counts,
    csrnet_sector_counts
):

    hybrid_counts = (
        create_sector_dict(
            0.0
        )
    )

    disagreement = (
        create_sector_dict(
            0.0
        )
    )

    fusion_mode = (
        create_sector_dict(
            "YOLO ONLY"
        )
    )

    for sector in SECTORS:

        yolo_count = float(
            yolo_sector_counts[
                sector
            ]
        )

        csr_count = float(
            csrnet_sector_counts[
                sector
            ]
        )

        # ----------------------------------------------------
        # CSRNet not available yet
        # ----------------------------------------------------

        if csr_count <= 0.5:

            hybrid = yolo_count

            disagreement_value = 0.0

            mode = "YOLO ONLY"


        # ----------------------------------------------------
        # YOLO sees nothing
        # ----------------------------------------------------

        elif yolo_count <= 0.5:

            # Dense-crowd case:
            # trust CSRNet, but slightly conservatively.
            hybrid = (
                csr_count *
                0.90
            )

            disagreement_value = 100.0

            mode = "CSRNET"


        else:

            difference = abs(
                yolo_count -
                csr_count
            )

            disagreement_value = (
                difference /
                max(
                    yolo_count +
                    csr_count,
                    1.0
                )
            ) * 100.0

            # ------------------------------------------------
            # Models agree
            # ------------------------------------------------

            if disagreement_value < 25:

                hybrid = (
                    yolo_count * 0.55
                    +
                    csr_count * 0.45
                )

                mode = "AGREEMENT"


            # ------------------------------------------------
            # CSRNet substantially higher
            # -> likely occlusion / dense crowd
            # ------------------------------------------------

            elif csr_count > yolo_count:

                hybrid = (
                    yolo_count * 0.35
                    +
                    csr_count * 0.65
                )

                mode = "CSRNET DENSE"


            # ------------------------------------------------
            # YOLO substantially higher
            # ------------------------------------------------

            else:

                hybrid = (
                    yolo_count * 0.65
                    +
                    csr_count * 0.35
                )

                mode = "YOLO DOMINANT"


        hybrid_counts[
            sector
        ] = max(
            float(hybrid),
            0.0
        )

        disagreement[
            sector
        ] = min(
            float(disagreement_value),
            100.0
        )

        fusion_mode[
            sector
        ] = mode

    return (
        hybrid_counts,
        disagreement,
        fusion_mode
    )


# ============================================================
# LOCAL MOVEMENT
# ============================================================
#
# Instead of one movement number for the whole frame,
# calculate movement separately for A1-B4.
# ============================================================

def calculate_local_movement(
    frame
):

    global previous_gray

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # Keep processing manageable.
    gray = cv2.resize(
        gray,
        (640, 360)
    )

    if previous_gray is None:

        previous_gray = gray

        return create_sector_dict(
            0.0
        )

    flow = cv2.calcOpticalFlowFarneback(

        previous_gray,

        gray,

        None,

        0.5,

        3,

        15,

        3,

        5,

        1.2,

        0
    )

    magnitude, _ = (
        cv2.cartToPolar(
            flow[..., 0],
            flow[..., 1]
        )
    )

    movement = (
        create_sector_dict(
            0.0
        )
    )

    h, w = gray.shape

    for sector in SECTORS:

        x1, y1, x2, y2 = (
            get_sector_bounds(
                sector,
                w,
                h
            )
        )

        region = (
            magnitude[
                y1:y2,
                x1:x2
            ]
        )

        if region.size == 0:

            value = 0.0

        else:

            mean_motion = float(
                np.mean(region)
            )

            value = min(
                mean_motion * 20.0,
                100.0
            )

        movement[
            sector
        ] = value

    previous_gray = gray

    return movement


# ============================================================
# CROWD DENSITY SCORE
# ============================================================

def calculate_density_score(
    hybrid_count,
    width,
    height
):

    # Number of pixels in one sector.
    sector_area = (
        width *
        height /
        (
            GRID_COLS *
            GRID_ROWS
        )
    )

    if sector_area <= 0:

        return 0.0

    # People per 100,000 pixels.
    people_density = (
        hybrid_count /
        sector_area
    ) * 100000.0

    # Map:
    #
    # 0 density -> 0 score
    # WARNING threshold -> ~50
    # ALERT threshold -> 100
    #
    score = (
        people_density /
        DENSITY_ALERT
    ) * 100.0

    return min(
        max(score, 0.0),
        100.0
    )


# ============================================================
# SECTOR RISK
# ============================================================

def calculate_sector_risk(
    sector,
    hybrid_count,
    density_score,
    movement,
    disagreement
):

    # --------------------------------------------------------
    # CPI
    # --------------------------------------------------------
    #
    # Density is most important.
    #
    # Movement is second.
    #
    # Model disagreement contributes because it can indicate
    # difficult/occluded crowd conditions.
    #
    cpi = (

        density_score * 0.60

        +

        movement * 0.25

        +

        disagreement * 0.15
    )

    cpi = min(
        max(cpi, 0.0),
        100.0
    )


    # --------------------------------------------------------
    # ALERT
    # --------------------------------------------------------

    alert_condition = (

        hybrid_count >= 15.0

        or

        (
            density_score >= 70.0
            and
            movement >= 45.0
        )

        or

        cpi >= ALERT_SCORE
    )


    # --------------------------------------------------------
    # WARNING
    # --------------------------------------------------------

    warning_condition = (

        hybrid_count >= 8.0

        or

        density_score >= 50.0

        or

        (
            hybrid_count >=
            MIN_PEOPLE_FOR_MOVEMENT
            and
            movement >= 45.0
        )

        or

        cpi >= WARNING_SCORE
    )


    if alert_condition:

        status = "ALERT"

    elif warning_condition:

        status = "WARNING"

    else:

        status = "NORMAL"


    return (
        float(cpi),
        status
    )


# ============================================================
# SMOOTH SECTOR SCORE
# ============================================================

def smooth_sector_score(
    sector,
    score
):

    old = sector_score_ema[
        sector
    ]

    # 70% previous
    # 30% current
    #
    # This prevents rapid flickering.
    smoothed = (
        old * 0.70
        +
        score * 0.30
    )

    sector_score_ema[
        sector
    ] = smoothed

    return smoothed


# ============================================================
# PROCESS FRAME
# ============================================================

def process_frame(
    frame
):

    global csrnet_counter
    global last_csrnet_count
    global last_csrnet_density_map


    # ========================================================
    # CLEAN FRAME
    # ========================================================

    original_frame = (
        frame.copy()
    )

    height, width = (
        frame.shape[:2]
    )


    # ========================================================
    # YOLO SECTOR COUNTS
    # ========================================================

    yolo_sector_counts = (
        create_sector_dict(
            0
        )
    )

    sector_boxes = (
        create_sector_boxes()
    )


    # ========================================================
    # YOLO
    # ========================================================

    results = yolo_model(

        original_frame,

        imgsz=YOLO_IMGSZ,

        conf=YOLO_CONF,

        classes=[0],

        verbose=False
    )


    # ========================================================
    # PROCESS YOLO PERSONS
    # ========================================================

    for result in results:

        if result.boxes is None:

            continue

        for box in result.boxes:

            confidence = float(
                box.conf[0]
            )

            if confidence < YOLO_CONF:

                continue


            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )


            # Safety clipping
            x1 = max(
                0,
                min(width - 1, x1)
            )

            x2 = max(
                0,
                min(width - 1, x2)
            )

            y1 = max(
                0,
                min(height - 1, y1)
            )

            y2 = max(
                0,
                min(height - 1, y2)
            )


            # ------------------------------------------------
            # Use bottom-center.
            #
            # This is better than box center for deciding
            # which ground/location sector the person occupies.
            # ------------------------------------------------

            center_x = (
                x1 + x2
            ) // 2

            foot_y = y2


            sector = get_sector(

                center_x,

                foot_y,

                width,

                height
            )


            yolo_sector_counts[
                sector
            ] += 1


            sector_boxes[
                sector
            ].append(

                (
                    x1,
                    y1,
                    x2,
                    y2
                )
            )


            # ------------------------------------------------
            # Draw YOLO detection
            # ------------------------------------------------

            cv2.rectangle(

                frame,

                (x1, y1),

                (x2, y2),

                (0, 255, 0),

                1,

                cv2.LINE_AA
            )


            cv2.circle(

                frame,

                (
                    center_x,
                    foot_y
                ),

                3,

                (255, 0, 0),

                -1
            )


    # ========================================================
    # YOLO TOTAL
    # ========================================================

    yolo_count = int(
        sum(
            yolo_sector_counts.values()
        )
    )


    # ========================================================
    # CSRNET
    #
    # First frame and then every CSRNET_INTERVAL frames.
    # ========================================================

    run_csrnet = (

        last_csrnet_density_map is None

        or

        csrnet_counter >=
        CSRNET_INTERVAL
    )


    if run_csrnet:

        try:

            (
                csrnet_count,
                density_map
            ) = calculate_csrnet_count(

                original_frame
            )

            last_csrnet_count = (
                csrnet_count
            )

            last_csrnet_density_map = (
                density_map
            )

        except Exception as e:

            print(
                "\nCSRNet error:",
                e
            )

        csrnet_counter = 0


    else:

        csrnet_count = (
            last_csrnet_count
        )


    csrnet_counter += 1


    # ========================================================
    # CSRNET SECTOR COUNTS
    # ========================================================

    csrnet_sector_counts = (
        density_map_to_sector_counts(

            last_csrnet_density_map
        )
    )


    # ========================================================
    # FUSE PER SECTOR
    # ========================================================

    (
        hybrid_sector_counts,
        sector_disagreement,
        fusion_mode
    ) = fuse_sector_counts(

        yolo_sector_counts,

        csrnet_sector_counts
    )


    # ========================================================
    # HYBRID TOTAL
    # ========================================================

    hybrid_count = float(
        sum(
            hybrid_sector_counts.values()
        )
    )


    # ========================================================
    # LOCAL MOVEMENT
    # ========================================================

    local_movement = (
        calculate_local_movement(
            original_frame
        )
    )


    # ========================================================
    # SECTOR RISK
    # ========================================================

    sector_density_score = (
        create_sector_dict(
            0.0
        )
    )

    sector_cpi = (
        create_sector_dict(
            0.0
        )
    )

    sector_risk = (
        create_sector_dict(
            "NORMAL"
        )
    )


    # ========================================================
    # PROCESS EVERY SECTOR
    # ========================================================

    for sector in SECTORS:

        hybrid_value = (
            hybrid_sector_counts[
                sector
            ]
        )

        movement_value = (
            local_movement[
                sector
            ]
        )

        disagreement_value = (
            sector_disagreement[
                sector
            ]
        )


        # ----------------------------------------------------
        # Density score
        # ----------------------------------------------------

        density_score = (
            calculate_density_score(

                hybrid_value,

                width,

                height
            )
        )


        sector_density_score[
            sector
        ] = density_score


        # ----------------------------------------------------
        # Local risk
        # ----------------------------------------------------

        cpi, status = (
            calculate_sector_risk(

                sector,

                hybrid_value,

                density_score,

                movement_value,

                disagreement_value
            )
        )


        # ----------------------------------------------------
        # Smooth score
        # ----------------------------------------------------

        smoothed_cpi = (
            smooth_sector_score(

                sector,

                cpi
            )
        )


        sector_cpi[
            sector
        ] = smoothed_cpi


        # ----------------------------------------------------
        # Re-evaluate status using smoothed score
        # ----------------------------------------------------

        if (

            smoothed_cpi >=
            ALERT_SCORE

            and

            hybrid_value >=
            8.0
        ):

            status = "ALERT"


        elif (

            smoothed_cpi >=
            WARNING_SCORE

            and

            hybrid_value >=
            3.0
        ):

            status = "WARNING"


        else:

            status = "NORMAL"


        sector_risk[
            sector
        ] = status


    # ========================================================
    # IMPORTANT:
    #
    # Select danger sector from actual sector risk.
    # ========================================================

    alert_sectors = [

        sector

        for sector in SECTORS

        if sector_risk[sector]
        == "ALERT"
    ]


    warning_sectors = [

        sector

        for sector in SECTORS

        if sector_risk[sector]
        == "WARNING"
    ]


    if alert_sectors:

        danger_sector = max(

            alert_sectors,

            key=lambda s:
                sector_cpi[s]
        )


    elif warning_sectors:

        danger_sector = max(

            warning_sectors,

            key=lambda s:
                sector_cpi[s]
        )


    else:

        danger_sector = "NONE"


    # ========================================================
    # HIGHEST SECTOR
    # ========================================================

    highest_sector = max(

        SECTORS,

        key=lambda s:
            sector_cpi[s]
    )


    highest_cpi = float(
        sector_cpi[
            highest_sector
        ]
    )


    # ========================================================
    # OVERALL STATUS
    # ========================================================

    if alert_sectors:

        overall_risk = "ALERT"

    elif warning_sectors:

        overall_risk = "WARNING"

    else:

        overall_risk = "NORMAL"


    # ========================================================
    # HIGHEST DENSITY
    # ========================================================

    highest_density_sector = max(

        SECTORS,

        key=lambda s:
            hybrid_sector_counts[s]
    )


    highest_density = float(

        hybrid_sector_counts[
            highest_density_sector
        ]
    )


    # ========================================================
    # WHOLE-FRAME HYBRID DENSITY
    # ========================================================

    scene_area = (
        width *
        height
    )

    if scene_area > 0:

        hybrid_density = (

            hybrid_count /
            scene_area
        ) * 100000.0

    else:

        hybrid_density = 0.0


    # ========================================================
    # RETURN
    # ========================================================

    return {

        # ----------------------------------------------------
        # Frame
        # ----------------------------------------------------

        "frame": frame,


        # ----------------------------------------------------
        # Total counts
        # ----------------------------------------------------

        "yolo_count":
            yolo_count,

        "csrnet_count":
            csrnet_count,

        "hybrid_count":
            hybrid_count,

        "total_people":
            round(
                hybrid_count
            ),


        # ----------------------------------------------------
        # Per-sector model counts
        # ----------------------------------------------------

        "yolo_sector_counts":
            yolo_sector_counts,

        "csrnet_sector_counts":
            csrnet_sector_counts,

        "hybrid_sector_counts":
            hybrid_sector_counts,


        # ----------------------------------------------------
        # Boxes
        # ----------------------------------------------------

        "sector_boxes":
            sector_boxes,


        # ----------------------------------------------------
        # Density
        # ----------------------------------------------------

        "sector_density":
            sector_density_score,

        "sector_density_score":
            sector_density_score,


        "hybrid_density":
            hybrid_density,


        # ----------------------------------------------------
        # Movement
        # ----------------------------------------------------

        "sector_movement":
            local_movement,

        "movement":
            max(
                local_movement.values()
            ),

        "movement_score":
            max(
                local_movement.values()
            ),


        # ----------------------------------------------------
        # Disagreement
        # ----------------------------------------------------

        "sector_disagreement":
            sector_disagreement,


        "model_disagreement":
            max(
                sector_disagreement.values()
            ),


        # ----------------------------------------------------
        # Fusion
        # ----------------------------------------------------

        "fusion_mode":
            fusion_mode,


        # ----------------------------------------------------
        # CPI
        # ----------------------------------------------------

        "sector_cpi":
            sector_cpi,

        "highest_cpi":
            highest_cpi,

        "cpi":
            highest_cpi,


        # ----------------------------------------------------
        # Risk
        # ----------------------------------------------------

        "sector_risk":
            sector_risk,

        "overall_risk":
            overall_risk,

        "risk_level":
            overall_risk,


        # ----------------------------------------------------
        # Danger sector
        # ----------------------------------------------------

        "danger_sector":
            danger_sector,

        "danger_score":
            highest_cpi,


        # ----------------------------------------------------
        # Highest density
        # ----------------------------------------------------

        "highest_density_sector":
            highest_density_sector,

        "highest_density":
            highest_density,


        # ----------------------------------------------------
        # Other
        # ----------------------------------------------------

        "highest_sector":
            highest_sector
    }