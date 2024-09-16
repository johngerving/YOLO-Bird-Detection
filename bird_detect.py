from ultralytics import YOLO
import cv2
import torch
from concurrent.futures import ProcessPoolExecutor
import numpy as np

FILE_NAME = 'footage/bird_footage.mp4'
GPU_COUNT = torch.cuda.device_count()

def process_video(gpu_num, file_name, start_index, stop_index):
    '''Processes a subset of frames in a video.

    :param gpu_num: The index of the CUDA device to run inference on.
    :param file_name: The path to the video to read from.
    :param start_index: The frame index to start from.
    :param stop_index: The frame index to stop at. The frames will be read up to, but not including, this index.
    '''

    # Initialize model and device to use for processing
    device = torch.device(f'cuda:{gpu_num}')
    model = YOLO('models/yolov10l.pt')
    model.to(device)    

    # Create a VideoCapture object and read from the file
    cap = cv2.VideoCapture(file_name)

    # Check if opened successfully
    if not cap.isOpened():
        raise Exception("Error opening video file")

    # Read frames until completed
    frame_number = start_index
    # Set the frame index to read from
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)

    # Create numpy array with detections from each frame
    detections = np.full(stop_index - start_index, False, dtype=bool)

    # Read frames in video
    while frame_number < stop_index and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            raise exception(f'Error: Frame {frame_number} could not be read.')

        # Run inference
        results = model.predict(frame, device=device, verbose=False)

        # Check if an object was detected in the frame
        object_detected = results[0].boxes.cls.size(dim=0) > 0
        # If it was, set the detection value to true
        if object_detected:
            detections[frame_number - start_index] = True   
            
        frame_number += 1

    print(detections)
    return detections
        

def total_video_frames(file_name):
    '''Returns the total number of frames in a video.

    :param file_name: The path to a video.

    :return: An integer representing the number of frames in the video.
    '''

    cap = cv2.VideoCapture(file_name)
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

def main():
    if GPU_COUNT < 1:
        print("Error: No GPUs found on system.")
        exit(1)

    devices = []
    file_names = []
    start_indices = []
    end_indices = []
    # Get the number of frames in the video
    num_frames = total_video_frames(FILE_NAME)

    # Split up the frames between the GPU
    batch_size = 1024 # num_frames // GPU_COUNT
    start = 0
    end = 0

    batch_num = 0
    while end < num_frames:
        # Add the batch size to the current end index
        end = start + batch_size

        # Extend to the end of the frame count if the next batch won't be a full batch
        if(end + batch_size > num_frames):
            end = num_frames

        # Add arguments for batch to list
        devices.append(batch_num % GPU_COUNT)
        file_names.append(FILE_NAME)
        start_indices.append(start)
        end_indices.append(end)

        # Set the start for the next loop
        start = end

        batch_num += 1

    num_batches = batch_num
        
    with ProcessPoolExecutor(max_workers=num_batches) as executor:
        executor.map(process_video, devices, file_names, start_indices, end_indices)

if __name__ == '__main__':
    main()