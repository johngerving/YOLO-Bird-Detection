from ultralytics import YOLO
import cv2
import torch
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from pathlib import Path

FOLDER = 'footage'
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

    return detections
        

def total_video_frames(file_name):
    '''Returns the total number of frames in a video.

    :param file_name: The path to a video.

    :return: An integer representing the number of frames in the video.
    '''

    cap = cv2.VideoCapture(file_name)

    # Check if opened successfully
    if not cap.isOpened():
        raise Exception(f"Error opening file {file_name}")
        
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

def calculate_clip_bounds(batches, padding=90):
    '''Calculates the start and end indices of the clips where detections occurred.

    :param batches: A generator object, with each entry containing a NumPy boolean array, each entry corresponding to a frame and True representing a detection occurring.
    :param padding: The number of frames before and after the detection to include as padding in the clips.

    :return: A two-dimensional list, with each element having two elements representing the start and end frame index of a clip. 
    '''

    currently_reading_clip = False
    frame_index = 0
    frames_since_last_detection = 0

    clips = []

    # Loop through batches
    for batch in batches:
        # Loop through each frame
        for frame in batch:
            if frame == True:
                # Reset frames since last detection if there was a detection in the frame
                frames_since_last_detection = 0

                # If a clip isn't already being read, add a new entry to the clips with the current frame index
                if not currently_reading_clip:
                    currently_reading_clip = True
                    start = max(frame_index - padding, 0)
                    clips.append([start])
            else:
                # Update frames since last detection if no detection in frame
                frames_since_last_detection += 1
                
                # If there is a clip currently being read and the frames since the last detection exceeds the padding, end the current clip and reset variables
                if currently_reading_clip and frames_since_last_detection > padding:
                    end = frame_index

                    clips[-1].append(end)

                    currently_reading_clip = False
                    frames_since_last_detection = 0

            frame_index += 1

    return clips

def read_and_write_clips(file_name, clips, output_dir):
    '''Reads a file from a path name and writes clips from it to new files.
    
    :param file_name: The path to the original file to read clips from.
    :param clips: An array of clips with start and end indices to read from.
    :param output_dir: The directory to write clips to.
    '''

    cap = cv2.VideoCapture(file_name)

    # Check if opened successfully
    if cap.isOpened() == False:
        raise Exception(f"Error opening video file {file_name}")

    # Get width and height of frame
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (frame_width, frame_height)
    # Get FPS of video
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Create output folder if it doesn't exist
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Loop through clips
    for i in range(len(clips)):
        # Write to MP4 format
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # Create new video writer for clip
        video = cv2.VideoWriter(str(output / f'{i}.mp4'), fourcc, fps, frame_size)

        # Set the current read frame to the start index of the clip
        cap.set(cv2.CAP_PROP_POS_FRAMES, clips[i][0] - 1)

        # Set current frame to start index of the clip
        current_frame = clips[i][0]

        # Loop through the clip until it is closed or the end index of the clip is reached
        while cap.isOpened() and current_frame <= clips[i][1]:
            # Read each frame of video
            ret, frame = cap.read()

            if ret == True:
                # Write the frame to the output file
                video.write(frame)
            else:
                break

            current_frame += 1

        video.release()

    cap.release()
    

def main():
    if GPU_COUNT < 1:
        print("Error: No GPUs found on system.")
        exit(1)

    dir = Path(FOLDER)

    for file in dir.glob('**/*.MOV'):
        print(f"Processing {str(file)}")
        devices = []
        file_names = []
        start_indices = []
        end_indices = []
        # Get the number of frames in the video
        num_frames = total_video_frames(str(file))
    
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
            file_names.append(str(file))
            start_indices.append(start)
            end_indices.append(end)
    
            # Set the start for the next loop
            start = end
    
            batch_num += 1
    
        num_batches = batch_num
            
        with ProcessPoolExecutor(max_workers=32) as executor:
            frame_detections = executor.map(process_video, devices, file_names, start_indices, end_indices)
    
            clips = calculate_clip_bounds(frame_detections)
            print(f"Writing to output/{file.stem}")
            read_and_write_clips(str(file), clips, f"output/{file.stem}")

if __name__ == '__main__':
    main()
