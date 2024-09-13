from ultralytics import YOLO
import cv2
import threading
import queue
import torch


BUFFER_SIZE = 16
GPU_COUNT = torch.cuda.device_count()

q = queue.Queue()

def process_frames(gpu_num):
    model = YOLO("yolov10l.pt")
    
    while True:
        frames = q.get()
        results = model.predict(frames)
        # print(f"Got results on thread {gpu_num}")
        q.task_done()
    
def read_frames(file_name):
    cap = cv2.VideoCapture(file_name)

    if(cap.isOpened() == False):
        raise Exception("Error opening video file")

    threads = [threading.Thread(target=process_frames, args=(i,), daemon=True) for i in range(GPU_COUNT)]

    print(threads)

    for thread in threads:
        thread.start()

    frames = []

    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
    
        if ret == True:
            while not q.empty():
                continue
            frames.append(frame)
            
            if(len(frames) == BUFFER_SIZE):
                q.put(frames)
                print(q.qsize())
                frames = []
        else:
            break

    q.join()


read_frames("bird_footage.mp4")

        