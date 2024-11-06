from ultralytics import YOLO
import cv2
import torch
from concurrent.futures import ProcessPoolExecutor
import threading
import queue
import numpy as np
from pathlib import Path
import sys
import logging

FOLDER = 'footage'
GPU_COUNT = torch.cuda.device_count()

log_formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s", 
    style="%",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger()

def process_video(gpu_num, file_name, start_index, stop_index, filters=None):
    '''Processes a subset of frames in a video.

    :param gpu_num: The index of the CUDA device to run inference on.
    :param file_name: The path to the video to read from.
    :param start_index: The frame index to start from.
    :param stop_index: The frame index to stop at. The frames will be read up to, but not including, this index.
    :param filters: A list of class names to filter. If none, all classes will be detected.
    '''

    # Initialize model and device to use for processing
    device = torch.device(f'cuda:{gpu_num}')
    model = YOLO('models/yolov10l.pt')
    model.to(device)   

    classes = None
    if filters is not None:
        classes_dict = model.names
        classes = []

        for name in filters:
            classes.append(list(classes_dict.keys())[list(classes_dict.values()).index(name)])

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
            logger.error(f'Frame {frame_number} number could not be read')
            raise Exception(f'frame {frame_number} could not be read.')

        # Run inference
        results = model.predict(frame, device=device, classes=classes, verbose=False)

        # Check if an object was detected in the frame
        object_detected = results[0].boxes.cls.size(dim=0) > 0
        # If it was, set the detection value to true
        if object_detected:
            detections[frame_number - start_index] = True   

            # image = results[0].orig_img

            # objects = []
            # for cls in results[0].boxes.cls.cpu().detach().numpy():
            #     objects.append(results[0].names[cls])
            
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

def calculate_clip_bounds(batches, num_frames, padding=90):
    '''Calculates the start and end indices of the clips where detections occurred.

    :param batches: A generator object, with each entry containing a NumPy boolean array, each entry corresponding to a frame and True representing a detection occurring.
    :param num_frames: The total number of frames in the video.
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

    if len(clips) > 0 and len(clips[-1]) < 2:
        clips[-1].append(num_frames - 1)

    return clips

def read_and_write_clip(file_name, clip, output_path):
    '''
    Writes a single clip from an input video to a new video.

    :param file_name: The path to an input video.
    :param clip: A list of length 2 containing the start and end frames of the clip.
    :param output_path: The path to write the output clip to.
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

    # Write to MP4 format
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Create new video writer for clip
    video = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    # Set the current read frame to the start index of the clip
    cap.set(cv2.CAP_PROP_POS_FRAMES, clip[0] - 1)

    # Set current frame to start index of the clip
    current_frame = clip[0]

    # Loop through the clip until it is closed or the end index of the clip is reached
    while cap.isOpened() and current_frame <= clip[1]:
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

def read_and_write_clips(file_name, clips, output_dir):
    '''
    Writes a list of clips to files.

    :param file_name: The path to the original file to read clips from.
    :param clips: An array of clips with start and end indices to read from.
    :param output_dir: The directory to write clips to.
    '''
    # Create output folder if it doesn't exist
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(max_workers=5) as e:
        # Loop through clips
        for i in range(len(clips)):
            e.submit(read_and_write_clip, file_name, clips[i], str(Path(output_dir) / f"{i}.mp4"))

q = queue.Queue()

def file_writer():
    '''
    A worker queue for writing video clips to the disk.
    '''

    while True:
        # Get latest item from queue
        item = q.get()

        # Write file to queue
        logger.info(f"Writing to output/{item['file'].stem}")
        read_and_write_clips(str(item['file']), item['clips'], f"output/{item['file'].stem}")
        logger.info(f"Finished writing output/{item['file'].stem}")

        # # Delete original file
        # logger.info(f"Deleting {item['file']}")
        # try:
        #     item['file'].unlink()
        #     logger.info(f"Finished deleting {item['file']}")
        # except Exception as e:
        #     logger.error(f"Could not delete {item['file']}: {e}")

        q.task_done()

def get_file_list(list_path):
    '''
    get_file_list returns a list of files given a path to a text file containing a path to a file on each line.

    :param list_path: The path to a text file containing a path to a file on each line.
    
    :return: A list of Path objects
    '''

    files = []

    with open(list_path) as f:
        # Get each line
        files_str = f.read().splitlines()

        for file in files_str:
            path = Path(file)

            # Make sure path exists and isn't a directory
            if path.exists() and not path.is_dir():
                files.append(path)

    return files
    

def main():
    logger.setLevel(logging.INFO)

    file_log_handler = logging.FileHandler("bird_detect.log")
    file_log_handler.setFormatter(log_formatter)

    console_log_handler = logging.StreamHandler()
    console_log_handler.setFormatter(log_formatter)

    logger.addHandler(file_log_handler)
    logger.addHandler(console_log_handler)

    logger.info('Started')

    if GPU_COUNT < 1:
        logger.error("No GPUs found on system.")
        exit(1)
    if len(sys.argv) < 2:
        logger.error("Missing file list argument.")
        exit(1)

    # Get name of file with list of videos to process
    f = Path(sys.argv[1])
    if not f.exists() or f.is_dir():
        logger.error(f"Invalid file {f}")
        exit(1)

    video_list = get_file_list(str(f))
    logger.info(f"Processing videos: {', '.join(map(str, video_list))}")

    # Create worker thread
    threading.Thread(target=file_writer, daemon=True).start()

    for file in video_list:
        logger.info(f"Processing {str(file)}")
        devices = []
        file_names = []
        start_indices = []
        end_indices = []
        filters = []
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
            filters.append(['bird'])
    
            # Set the start for the next loop
            start = end
    
            batch_num += 1
    
        num_batches = batch_num
            
        with ProcessPoolExecutor(max_workers=32) as executor:
            frame_detections = executor.map(process_video, devices, file_names, start_indices, end_indices, filters)

            clips = calculate_clip_bounds(frame_detections, num_frames)

            if len(clips) == 0:
                logger.info(f"No clips found in {file}")

            logger.info(f"Finished processing {file}")

            # Send clip information to queue
            q.put({
                "file": file,
                "clips": clips,
            })

    q.join()
    logger.info('Finished')
            

if __name__ == '__main__':
    main()
