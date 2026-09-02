import cv2
import time

input_path = "C:\\dl_project\\project\\data\\input\\road_video.mp4"
output_path = "C:\\dl_project\\project\\data\\output\\processed_video.mp4"

cap = cv2.VideoCapture(input_path)

if not cap.isOpened():
    print("Error: Could not open input video.")
    exit()

# Original video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Resize dimensions
new_width = 1280
new_height = 720

# Video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (new_width, new_height)
)

frame_number = 0

start_time = time.time()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # Resize frame
    frame = cv2.resize(
        frame,
        (new_width, new_height)
    )

    # Calculate elapsed time
    elapsed_time = time.time() - start_time

    current_fps = (
        frame_number / elapsed_time
        if elapsed_time > 0
        else 0
    )

    # Add information to frame
    cv2.putText(
        frame,
        f"Frame: {frame_number}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"FPS: {current_fps:.2f}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Display
    cv2.imshow(
        "Road Safety - OpenCV Pipeline",
        frame
    )

    # Save processed frame
    out.write(frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Processing completed.")
print(f"Processed frames: {frame_number}")
print(f"Output saved to: {output_path}")