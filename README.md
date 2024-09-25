# YOLO Bird Detection

A Python script for detecting the presence of birds in footage. The model does not detect the presence of birds specifically, but rather any object the model detects.

## Getting started

Clone the repository with the following:
```
git clone https://gitlab.nrp-nautilus.io/humboldt/yolo-bird-detection.git
```

Place a folder called "footage/" in the repository directory. It should contain .MOV files with the footage to be processed.

Run the script with the following:

On Linux:
```
python3 bird_detect.py
```

On Windows:
```
py bird_detect.py
```

Once the script is finished, there will be a directory called "output/" in the repository directory. Inside, there will be a folder for each video processed, each of which will contain a clip in which an object was detected.

## Program overview

The program begins by checking if CUDA is supported on the machine. If it is not, it exits with an error.

Then, the script finds all .MOV files in the "footage/" directory. For each video, it does the following:
1. Split the video into batches of 1024 frames.
2. Start a process for each batch. Assign a GPU to each batch.
3. Run the model on each frame in each batch. If an object was detected, write a 1 to an array of frame detections, otherwise write 0.
4. Using the list of frame detections, create a list of clip starting and ending frames, each clip containing an instance of an object being detected.
5. Read the frames in each clip and write them to a new file.
