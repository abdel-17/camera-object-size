# Camera Object Size
Capture objects in video and calculate their size.

## Libraries
1. Numpy
2. OpenCV

## File Structure
You'll mainly run `main.py`, which capture videos from the
webcam and calculates the size of each object in each frame.

`imutils.py` contains the logic for processing each frame,
finding contours in those frames and their dimensions, etc.

`test.ipynb` is a Jupyter notebook for testing the output
of the functions in the `imutils` module.

Tested on Python 3.11, but should work on versions below that.
