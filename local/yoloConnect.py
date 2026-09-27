import os
import requests
import tkinter as tk
import cv2
from tkinter import filedialog
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import webbrowser
import json

#pip freeze > requirements.txt use for generating local requirements.txt file for reproducibility.

weights_url = hf_hub_download(repo_id="Hanishka-27/waste-classification-yolov8-ken", filename="yolov8n-waste-12cls-best.pt")

model = YOLO(weights_url)

root = tk.Tk()
root.title("PollutPonder")
root.geometry("400x300")


#Colab assisted
def getFrames(video_path):      
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = video.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        video.release()
        raise ValueError("Could not determine the video's frame rate")

    frame_interval = max(1, round(fps * 5))
    frame_number = 0
    saved_count = 0
    output_dir = "local/frames"
    os.makedirs(output_dir, exist_ok=True)

    try:
        while True:
            success, frame = video.read()
            if not success:
                break

            if frame_number % frame_interval == 0:
                output_path = os.path.join(output_dir, f"frame_{saved_count:04}.jpg")
                cv2.imwrite(output_path, frame)  # raw frame — no model call here
                saved_count += 1

            frame_number += 1
    finally:
        video.release()


#Colab assisted
def analyzeFrames():
    input_dir = "local/frames"
    annotated_dir = "local/yolo/analyzed_frames"
    coords_dir = "local/yolo/coordinates"

    os.makedirs(annotated_dir, exist_ok=True)
    os.makedirs(coords_dir, exist_ok=True)

    frame_files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".png")))

    for filename in frame_files:
        print(f"Processing {filename}...")
        frame_path = os.path.join(input_dir, filename)
        frame = cv2.imread(frame_path)

        if frame is None:
            print(f"Could not read {frame_path}, skipping")
            continue

        result = model(frame, verbose=False)[0]

        annotated_frame = result.plot()
        cv2.imwrite(os.path.join(annotated_dir, filename), annotated_frame)

        detections = []
        for box, score, cls in zip(
            result.boxes.xyxy.tolist(),
            result.boxes.conf.tolist(),
            result.boxes.cls.tolist(),
        ):
            detections.append({
                "label": result.names[int(cls)],
                "score": round(float(score), 4),
                "box": [round(v, 1) for v in box],
            })

        json_filename = os.path.splitext(filename)[0] + ".json"
        with open(os.path.join(coords_dir, json_filename), "w") as f:
            json.dump(detections, f, indent=2)

        print(f"{filename}: {len(result.boxes)} boxes detected")

def fileUpload():
    path = filedialog.askopenfilename(title="Select Video", 
                                      filetypes=[("Video Files", "*.mp4 *.avi *.mov")],
                                      initialdir=f"C:/Users/{os.getlogin()}/Videos")
    getFrames(path)
    analyzeFrames()

estimatePollButton = tk.Button(root, text="Estimate Pollution", 
                               command=lambda: webbrowser.open("siteEstimatePage.com"))
detectPollButton = tk.Button(root, text="Detect Pollution", 
                             command=lambda: fileUpload())

estimatePollButton.pack(pady=10, padx=10)
detectPollButton.pack(pady=10, padx=10)

root.mainloop()