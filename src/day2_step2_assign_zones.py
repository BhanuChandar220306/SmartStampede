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

    # Get frame dimensions
    height, width = frame.shape[:2]

    # -----------------------------
    # Create 3 zones
    # -----------------------------

    zone_height = height // 3

    y1 = zone_height
    y2 = zone_height * 2

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

    # Zone names
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
    # Process detected people
    # -----------------------------

    for result in results:

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            # Only person
            if class_id != 0:
                continue

            # Ignore very low-confidence detections
            if confidence < 0.40:
                continue

            # Bounding box
            x1_box, y1_box, x2_box, y2_box = map(
                int,
                box.xyxy[0]
            )

            # -----------------------------
            # Find center of person
            # -----------------------------

            center_x = (x1_box + x2_box) // 2
            center_y = (y1_box + y2_box) // 2

            # -----------------------------
            # Determine zone
            # -----------------------------

            if center_y < y1:

                zone = "ZONE A"

            elif center_y < y2:

                zone = "ZONE B"

            else:

                zone = "ZONE C"

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

            # Draw center point
            cv2.circle(
                frame,
                (center_x, center_y),
                4,
                (255, 0, 0),
                -1
            )

            # Display zone above person
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
    # Display
    # -----------------------------

    cv2.imshow(
        "Day 2 - People Assigned to Zones",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()