import cv2

# -----------------------------
# VIDEO
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

    # --------------------------------
    # Divide frame into 3 horizontal zones
    # --------------------------------

    zone_height = height // 3

    # Zone boundaries
    y1 = zone_height
    y2 = zone_height * 2

    # --------------------------------
    # Draw Zone A
    # --------------------------------

    cv2.line(
        frame,
        (0, y1),
        (width, y1),
        (0, 255, 0),
        3
    )

    cv2.putText(
        frame,
        "ZONE A",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # --------------------------------
    # Draw Zone B
    # --------------------------------

    cv2.line(
        frame,
        (0, y2),
        (width, y2),
        (0, 255, 255),
        3
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

    # --------------------------------
    # Draw Zone C
    # --------------------------------

    cv2.putText(
        frame,
        "ZONE C",
        (20, y2 + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    # Display
    cv2.imshow(
        "Day 2 - Zone Division",
        frame
    )

    # Press Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()