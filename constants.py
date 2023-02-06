import cv2

# The ASCII code for the escape character.
ESCAPE = 27

# Define the width and height of A4 paper.
# All measurements are performed with respect to its size.
PAPER_SCALE = 20
A4_WIDTH = int(21.0 * PAPER_SCALE)
A4_HEIGHT = int(29.7 * PAPER_SCALE)

# Define some colors as BGR values.
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Define the line properties.
LINE_COLOR = GREEN
LINE_THICKNESS = 3

# Define the arrow properties.
ARROW_COLOR = RED
ARROW_THICKNESS = 3

# Define the text properties.
FONT = cv2.FONT_HERSHEY_COMPLEX
FONT_SCALE = 0.8
TEXT_COLOR = BLACK
TEXT_THICKNESS = 2
