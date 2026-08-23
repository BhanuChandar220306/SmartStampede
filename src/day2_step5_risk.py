import cv2
from ultralytics import YOLO

# -----------------------------
# Load YOLO model
# -----------------------------
model = YOLO("yolov8n.pt")

# -----------------------------
# Open video
# -----------------------------
video_path = "videos/crowd.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video")
    exit()

print("Video opened successfully")


# -----------------------------
# Zone capacities
# -----------------------------
ZONE_A_CAPACITY = 50
ZONE_B_CAPACITY = 50
ZONE_C_CAPACITY = 50


# -----------------------------
# Risk calculation function
# -----------------------------
def calculate_risk(density):

    if density < 40:
        return "LOW"

    elif density < 70:
        return "MEDIUM"

    elif density < 90:
        return "HIGH"

    else:
        return "CRITICAL"


while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Frame dimensions
    height, width = frame.shape[:2]

    # -----------------------------
    # Create zones
    # -----------------------------
    zone_height = height // 3

    y1 = zone_height
    y2 = zone_height * 2

    # -----------------------------
    # Reset counters
    # -----------------------------
    zone_a_count = 0
    zone_b_count = 0
    zone_c_count = 0

    # -----------------------------
    # Draw zone boundaries
    # -----------------------------

    cv2.line(
        frame,
        (0, y1),
        (width, y1),
        (0, 255, 0),
        3
    )

    cv2.line(
        frame,
        (0, y2),
        (width, y2),
        (0, 255, 255),
        3
    )

    # -----------------------------
    # Run YOLO
    # -----------------------------

    results = model(frame, verbose=False)

    # -----------------------------
    # Process people
    # -----------------------------

    for result in results:

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            # Person only
            if class_id != 0:
                continue

            if confidence < 0.40:
                continue

            # Bounding box
            x1_box, y1_box, x2_box, y2_box = map(
                int,
                box.xyxy[0]
            )

            # Person center
            center_x = (x1_box + x2_box) // 2
            center_y = (y1_box + y2_box) // 2

            # -----------------------------
            # Assign zone
            # -----------------------------

            if center_y < y1:

                zone = "ZONE A"
                zone_a_count += 1

            elif center_y < y2:

                zone = "ZONE B"
                zone_b_count += 1

            else:

                zone = "ZONE C"
                zone_c_count += 1

            # -----------------------------
            # Draw bounding box
            # -----------------------------

            cv2.rectangle(
                frame,
                (x1_box, y1_box),
                (x2_box, y2_box),
                (0, 255, 0),
                2
            )

            cv2.circle(
                frame,
                (center_x, center_y),
                4,
                (255, 0, 0),
                -1
            )

            cv2.putText(
                frame,
                zone,
                (x1_box, max(20, y1_box - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

    # ==================================================
    # STEP 4 - DENSITY
    # ==================================================

    density_a = (
        zone_a_count / ZONE_A_CAPACITY
    ) * 100

    density_b = (
        zone_b_count / ZONE_B_CAPACITY
    ) * 100

    density_c = (
        zone_c_count / ZONE_C_CAPACITY
    ) * 100

    # ==================================================
    # STEP 5 - RISK LEVEL
    # ==================================================

    risk_a = calculate_risk(density_a)
    risk_b = calculate_risk(density_b)
    risk_c = calculate_risk(density_c)

    # -----------------------------
    # Display Zone A
    # -----------------------------

    cv2.putText(
        frame,
        f"ZONE A: {zone_a_count} | "
        f"{density_a:.1f}% | {risk_a}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )

    # -----------------------------
    # Display Zone B
    # -----------------------------

    cv2.putText(
        frame,
        f"ZONE B: {zone_b_count} | "
        f"{density_b:.1f}% | {risk_b}",
        (20, y1 + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )

    # -----------------------------
    # Display Zone C
    # -----------------------------

    cv2.putText(
        frame,
        f"ZONE C: {zone_c_count} | "
        f"{density_c:.1f}% | {risk_c}",
        (20, y2 + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255),
        2
    )

    # -----------------------------
    # Show video
    # -----------------------------

    cv2.imshow(
        "Day 2 - Crowd Risk Detection",
        frame
    )

    # Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()