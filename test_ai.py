import cv2

from ai_engine import process_frame


video_path = "videos/crowd.mp4"

cap = cv2.VideoCapture(video_path)


if not cap.isOpened():

    print("Error opening video")
    exit()


while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Process frame
    data = process_frame(frame)

    # Get processed frame
    output_frame = data["frame"]

    # Print actual results
    print(
        "People:",
        data["total_people"],
        "| Zone A:",
        data["zone_a_count"],
        "| Zone B:",
        data["zone_b_count"],
        "| Zone C:",
        data["zone_c_count"],
        "| Risk:",
        data["overall_risk"]
    )

    # Display
    cv2.imshow(
        "AI Crowd Detection",
        output_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()