import cv2
import os
import signal
import sys
from pathlib import Path
from datetime import datetime

GST_PIPELINE = "libcamerasrc ! video/x-raw,width=640,height=480 ! videoflip method=rotate-180 ! videoscale ! video/x-raw,width=640,height=640 ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false"

FPS = 30
CLIP_DURATION_SECONDS = 60
FRAME_WIDTH = 640
FRAME_HEIGHT = 640

OUTPUT_DIR = Path("clips")

running = True
writer = None
cap = None


def cleanup_and_exit():
    global writer, cap

    print("\nStopping recording...")

    if writer is not None:
        writer.release()
        writer = None

    if cap is not None:
        cap.release()
        cap = None

    cv2.destroyAllWindows()

    print("Resources released successfully.")
    sys.exit(0)


def signal_handler(sig, frame):
    global running
    running = False


signal.signal(signal.SIGINT, signal_handler)


def get_next_clip_index():
    OUTPUT_DIR.mkdir(exist_ok=True)

    existing = []

    for file in OUTPUT_DIR.glob("*.mp4"):
        try:
            existing.append(int(file.stem))
        except ValueError:
            pass

    if not existing:
        return 1

    return max(existing) + 1


def main():
    global writer, cap, running

    clip_index = get_next_clip_index()

    cap = cv2.VideoCapture(GST_PIPELINE, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("Failed to open camera.")
        return

    print("Recording started.")
    print("Press Ctrl+C to stop safely.\n")

    frames_per_clip = FPS * CLIP_DURATION_SECONDS

    while running:

        output_path = OUTPUT_DIR / f"{clip_index}.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writer = cv2.VideoWriter(
            str(output_path), fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT)
        )

        if not writer.isOpened():
            print(f"Failed to create video file: {output_path}")
            break

        print(f"Recording clip {clip_index} -> {output_path}")

        frame_count = 0

        while frame_count < frames_per_clip and running:

            ret, frame = cap.read()

            if not ret:
                print("Failed to read frame.")
                running = False
                break

            writer.write(frame)

            frame_count += 1

        writer.release()
        writer = None

        print(f"Finished clip {clip_index}")

        clip_index += 1

    cleanup_and_exit()


if __name__ == "__main__":
    main()
