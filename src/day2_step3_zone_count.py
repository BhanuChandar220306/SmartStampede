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

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Frame dimensions
    height, width = frame.shape[:2]

    # -----------------------------
    # Create 3 zones
    # -----------------------------
    zone_height = height // 3

    y1 = zone_height
    y2 = zone_height * 2

    # -----------------------------
    # Reset counters EVERY FRAME
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

    # Zone labels
    cv2.putText(
        frame,
        "ZONE A",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "ZONE B",
        (20, y1 + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "ZONE C",
        (20, y2 + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
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

            # Only persons
            if class_id != 0:
                continue

            # Confidence filter
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
            # Determine zone
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

            # Center point
            cv2.circle(
                frame,
                (center_x, center_y),
                4,
                (255, 0, 0),
                -1
            )

            # Zone above person
            cv2.putText(
                frame,
                zone,
                (x1_box, max(20, y1_box - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

    # -----------------------------
    # Display zone counts
    # -----------------------------

    cv2.putText(
        frame,
        f"Zone A: {zone_a_count}",
        (width - 250, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Zone B: {zone_b_count}",
        (width - 250, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Zone C: {zone_c_count}",
        (width - 250, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    # Total
    total_people = (
        zone_a_count +
        zone_b_count +
        zone_c_count
    )

    cv2.putText(
        frame,
        f"Total: {total_people}",
        (width - 250, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # -----------------------------
    # Show video
    # -----------------------------

    cv2.imshow(
        "Day 2 - Zone People Counting",
        frame
    )

    # Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()