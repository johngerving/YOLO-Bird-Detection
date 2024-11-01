
from pathlib import Path
import time
import cv2
from concurrent.futures import ProcessPoolExecutor

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


def read_and_write_clip(file_name, clip, output_path):
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

def read_and_write_clips_new(file_name, clips, output_dir):
    # Create output folder if it doesn't exist
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(max_workers=8) as e:
        # Loop through clips
        for i in range(len(clips)):
            e.submit(read_and_write_clip, file_name, clips[i], str(Path(output_dir) / f"{i}.mp4"))

def main():
    start = time.perf_counter()
    read_and_write_clips('footage/20230704_071638.MOV', [[0, 2000], [8000, 10000]], "output")
    end = time.perf_counter()
    elapsed = end - start
    print(f'Base function: {elapsed:.3f} seconds')

    start = time.perf_counter()
    read_and_write_clips_new('footage/20230704_071638.MOV', [[0, 2000], [8000, 10000]], "output")
    end = time.perf_counter()
    elapsed = end - start
    print(f'Multiprocessed function: {elapsed:.3f} seconds')

if __name__ == '__main__':
    main()