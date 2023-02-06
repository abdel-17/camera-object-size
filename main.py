import cv2
import imutils
from constants import *


def handle_frame(frame: cv2.Mat, only_detect_rectangles: bool = True):
    # Show the current frame in a window called "Original".
    cv2.imshow("Original", frame)

    # Find the contours in the current frame with minimum area of 1000.
    contours = imutils.find_contours(image=frame, min_area=1000)
    # Find the largest rectangle to detect the A4 paper.
    rectangles = imutils.find_approximating_polygons(contours, filter_by_sides=4)
    if not rectangles:
        # Return early if we don't find any rectangles.
        return

    # Warp the image to the prespective of the A4, which is assumed to be the
    # largest rectangle. Remember that contours are sorted by decreasing area.
    paper_frame = imutils.warp_image(
        image=frame, rectangle=rectangles[0], width=A4_WIDTH, height=A4_HEIGHT
    )
    # Loop over each contour in the A4's frame to find the objects inside.
    min_area = 25 * PAPER_SCALE**2
    inner_contours = imutils.find_contours(image=paper_frame, min_area=min_area)

    if only_detect_rectangles:
        inner_rectangles = imutils.find_approximating_polygons(
            inner_contours, filter_by_sides=4
        )
    else:
        inner_rectangles = [
            imutils.find_min_area_rectangle(contour)
            for contour in inner_contours
        ]

    # Draw a rectangle around each object found.
    imutils.draw_contours(image=paper_frame, contours=inner_rectangles)

    # Draw the dimensions of each object found.
    for rectangle in inner_rectangles:
        imutils.draw_dimensions(image=paper_frame, rectangle=rectangle)

    cv2.imshow("Paper", paper_frame)


capture = cv2.VideoCapture(1)

while True:
    # Read from the camera. Two values are returned, the read frame
    # and a boolean indicating whether the operation was successful.
    successful, frame = capture.read()
    if not successful:
        print("Cannot read the frame. Exiting...")
        break

    # Passing `True` to `only_rectangles` increases measurement
    # accuracy, but it cannot detect non-rectangular shapes.
    handle_frame(frame, only_detect_rectangles=True)

    # Exit if the esc key is pressed.
    if cv2.waitKey(1) == ESCAPE:
        break

# Release control of the camera after we're done.
capture.release()
