import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import streamlit as st
import pickle
import pandas as pd

# Path configurations
DATASET_PATH = "urldata.csv"
MODEL_PATH = "trained_models.pkl"

class FileChangeHandler(FileSystemEventHandler):
    def __init__(self, reload_callback):
        self.reload_callback = reload_callback

    def on_modified(self, event):
        if event.src_path.endswith(DATASET_PATH) or event.src_path.endswith(MODEL_PATH):
            print(f"🔄 Detected change in {event.src_path}, reloading...")
            self.reload_callback()

def watch_files(reload_callback):
    event_handler = FileChangeHandler(reload_callback)
    observer = Observer()
    observer.schedule(event_handler, ".", recursive=True)  # Watch the current directory
    observer.start()
    try:
        while True:
            time.sleep(2)  # Adjust this if needed
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

def reload_models():
    global model, df
    print("📂 Reloading dataset and models...")

    # Load dataset
    df = pd.read_csv(DATASET_PATH)
    st.session_state["df"] = df

    # Load trained model
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
        st.session_state["model"] = model

if __name__ == "__main__":
    print("👀 Watching for changes in dataset or models...")
    watch_files(reload_models)
