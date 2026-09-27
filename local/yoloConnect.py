import os
from dotenv import load_dotenv
import requests
import tkinter as tk
import cv2
from tkinter import filedialog
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import json
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pypdf
from google import genai

#pip freeze > requirements.txt use for generating local requirements.txt file for reproducibility.

load_dotenv("C:/1HK/hf.env")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

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

def openSecondWindow():
    second_window = tk.Toplevel(root)
    second_window.title("Gemini")
    second_window.geometry("800x640")
    label = tk.Label(second_window, text="Gemini")
    label.pack(pady=20)
    show_graph()

def show_graph():
    counts = getTrashTonsFromPDFs()

    graph_win = tk.Toplevel(root)
    graph_win.title("Detection Summary")
    graph_win.geometry("500x400")

    fig = Figure(figsize=(5, 4), dpi=100)
    plot = fig.add_subplot(111)
    plot.bar(counts.keys(), counts.values())
    plot.set_title("Reports of Pollution (years 2021-2025)")
    plot.set_ylabel("Count (tons)")
    plot.tick_params(axis='x', rotation=45)

    canvas = FigureCanvasTkAgg(fig, master=graph_win)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

def extract_text_from_pdfs(folder_path="local/Papers/Datasets"):
    combined_text = ""
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            path = os.path.join(folder_path, filename)
            reader = pypdf.PdfReader(path)
            for page in reader.pages:
                combined_text += page.extract_text() or ""
            combined_text += f"\n\n--- end of {filename} ---\n\n"
    return combined_text

def show_graph_from_dict(data: dict, title="Data", ylabel="Count"):
    graph_win = tk.Toplevel(root)
    graph_win.title(title)
    graph_win.geometry("500x400")

    fig = Figure(figsize=(5, 4), dpi=100)
    plot = fig.add_subplot(111)

    labels = list(data.keys())
    values = list(data.values())

    plot.bar(labels, values)
    plot.set_title(title)
    plot.set_ylabel(ylabel)
    plot.tick_params(axis='x', rotation=45)
    fig.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=graph_win)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

def getTrashTonsFromPDFs(folder_path="local/Papers/Datasets"):
    pdf_text = extract_text_from_pdfs(folder_path)

    if not pdf_text.strip():
        print("No text found in PDFs.")
        return None

    prompt = (
        "The following text comes from one or more reports about trash/waste collection.\n\n"
        f"{pdf_text}\n\n"
        "Summarize the total trash collected, in TONS, broken down by category or area "
        "if the data supports it. Respond with ONLY a valid JSON object mapping each "
        "category/area name to a numeric ton value — no explanation, no markdown, "
        "no code fences."
    )

    response = client.models.generate_content(model="gemini-flash-latest", contents=prompt)
    text = response.text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        print("Gemini didn't return valid JSON:", text)
        return None

    return data

def openTrashTonsGraph():
    folder_path = filedialog.askdirectory(title="Select folder with PDF reports")
    if not folder_path:
        return

    data = getTrashTonsFromPDFs(folder_path)
    if data:
        show_graph_from_dict(data, title="Trash Collected (Tons)", ylabel="Tons")

    tonsButton = tk.Button(root, text="Summarize Trash Tons from PDFs", command=openTrashTonsGraph)
    tonsButton.pack(pady=10, padx=10)

estimatePollButton = tk.Button(root, text="Estimate Pollution", 
                               command=lambda: openSecondWindow())
detectPollButton = tk.Button(root, text="Detect Pollution", 
                             command=lambda: fileUpload())

estimatePollButton.pack(pady=10, padx=10)
detectPollButton.pack(pady=10, padx=10)

root.mainloop()