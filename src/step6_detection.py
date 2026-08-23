import cv2
from ultralytics import YOLO

# Load YOLO
model = YOLO("yolov8n.pt")

print("YOLO model loaded successfully")

# Open video
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

    # Run YOLO
    results = model(frame)

    person_count = 0

    # Process results
    for result in results:

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            # Person only
            if class_id == 0:

                person_count += 1

                print(
                    "Person detected:",
                    round(confidence, 2)
                )

    # Display current count
    cv2.putText(
        frame,
        f"Persons Detected: {person_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    cv2.imshow("YOLO Person Detection", frame)

    # Q = quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()