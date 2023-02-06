import cv2
import numpy as np
from constants import *


def preprocess_image(image) -> cv2.Mat:
    # Convert the image to grayscale first.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply a 5x5 Gaussian blur filter.
    blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=1)
    # Apply the Canny edge detection filter.
    canny = cv2.Canny(blurred, threshold1=100, threshold2=100)
    # Apply dilation and erodion with a 5x5 average kernel.
    kernel = np.ones(shape=(5, 5))
    dilated = cv2.dilate(canny, kernel, iterations=3)
    eroded = cv2.erode(dilated, kernel, iterations=2)
    return eroded


def find_contours(image, min_area: int = ...):
    contours, _ = cv2.findContours(
        # Apply aome preprocessing to better detect contours.
        image=preprocess_image(image),
        # Keep the outer edges.
        mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_SIMPLE
    )
    # Sort contours by their area in decreasing order.
    areas = [cv2.contourArea(contour) for contour in contours]
    if min_area is not ...:
        # Filter out contours whose area are less than the given minimum.
        contours = [
            contour
            for contour, area in zip(contours, areas)
            if area >= min_area
        ]
    # Sort contours by decreasing area.
    return [
        contour
        for contour, _ in sorted(
            zip(contours, areas),
            key=lambda pair: pair[1],
            reverse=True
        )
    ]


def draw_contours(image, contours):
    cv2.drawContours(
        image=image,
        contours=contours,
        # When the index is -1, it draws all contours.
        contourIdx=-1,
        color=LINE_COLOR,
        thickness=LINE_THICKNESS
    )


def find_approximating_polygons(contours, filter_by_sides: int = ...):
    # Generate the approximating polygons for each contour.
    # Use 10% of the perimeter as the error bound.
    polygons = [
        cv2.approxPolyDP(
            contour,
            epsilon=0.1 * cv2.arcLength(contour, closed=True),
            closed=True
        )
        for contour in contours
    ]
    # If we filter by the number of sides, filter out polygons
    # whose number of sides is not equal to the given number.
    if filter_by_sides is not ...:
        polygons = [
            polygon
            for polygon in polygons
            if len(polygon) == filter_by_sides
        ]
    return polygons


def find_min_area_rectangle(contour):
    rectangle = cv2.minAreaRect(contour)
    # Extract the points of the rectangle.
    box = cv2.boxPoints(rectangle)
    box = __reorder_vertices(box)
    # Convert the coordinates to integers.
    return np.array(box, dtype=np.int32)


def warp_image(image, rectangle, width, height):
    # Reorder the input point.
    rectangle = __reorder_vertices(rectangle)
    # The coordinates must be converted to float32
    # for `getPerspectiveTransform` to work.
    rectangle = np.array(rectangle, dtype=np.float32)
    vertices = [
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ]
    desired_rectangle = np.array(vertices, dtype=np.float32)
    # Transform the image, changing the perspective to that of the given rectangle.
    transform = cv2.getPerspectiveTransform(rectangle, desired_rectangle)
    image = cv2.warpPerspective(image, transform, dsize=(width, height))

    # Apply padding because the image isn't warped perfectly.
    padding = 10
    return image[padding: height - padding, padding: width - padding]


def __reorder_vertices(rectangle):
    """
    Reorders the given rectangle's vertices to the following order:
        - top-left
        - top-right
        - bottom-right
        - bottom-left.
    """
    # Rectangles returned from `approxPolyDP()` have a weird shape of (4, 1, 2).
    # The second dimension is redundant, so we remove it.
    rectangle = np.squeeze(rectangle)
    assert rectangle.shape == (4, 2)

    # Add the coordinates of each point.
    coordinates_sum = rectangle.sum(axis=1)
    # Subtract the coordinates of each point.
    coordinates_diff = np.diff(rectangle, axis=1)

    ordered_points = np.empty_like(rectangle)
    # The top-left vertex makes the smallest distance with the origin.
    ordered_points[0] = rectangle[np.argmin(coordinates_sum)]
    # The bottom-right vertex, however, makes the largest distance.
    ordered_points[2] = rectangle[np.argmax(coordinates_sum)]
    # The coordinates of the top-right vertex have the minimum difference.
    # This is because its y is at its minimum while its x is at its maximum,
    # so y - x is at its minimum value.
    ordered_points[1] = rectangle[np.argmin(coordinates_diff)]
    # For similar reasons, the coordinates of the bottom-left vertiex have
    # the maximum difference (y is at its maximum while x is at its minimum).
    ordered_points[3] = rectangle[np.argmax(coordinates_diff)]
    return ordered_points


def draw_dimensions(image, rectangle):
    # Reorder the rectangle's vertices. This is needed to
    # determine how to draw the arrowed lines.
    rectangle = __reorder_vertices(rectangle)

    # Draw two arrowed lines, one for each of the width and height.
    # The width arrow starts from the top-left corner ending at the top-right.
    __draw_arrowed_line(image, rectangle[0], rectangle[1])
    # The height arrow starts from the top-left corner ending at the bottom-left.
    __draw_arrowed_line(image, rectangle[0], rectangle[3])

    # Draw the width and height as text on the image.
    width, height = __calculate_dimensions(rectangle)

    text = f"{width:.1f} cm"
    text_width, text_height = __calculate_text_size(text=text)
    org = rectangle[1]
    # Alight text horizontally to the right of the arrow's end.
    org[0] -= text_width
    # Position the text above the horizontal arrow.
    org[1] -= text_height
    __draw_text(image, text, org)

    text = f"{height:.1f} cm"
    text_width, text_height = __calculate_text_size(text=text)
    org = rectangle[3]
    # Position the width text below the vertical arrow.
    org[1] += text_height
    __draw_text(image, text, org)


def __draw_arrowed_line(image, pt1, pt2):
    cv2.arrowedLine(
        img=image,
        pt1=pt1,
        pt2=pt2,
        color=ARROW_COLOR,
        thickness=ARROW_THICKNESS
    )


def __draw_text(image, text, org):
    cv2.putText(
        img=image,
        text=text,
        org=org,
        fontFace=FONT,
        fontScale=FONT_SCALE,
        color=TEXT_COLOR,
        thickness=TEXT_THICKNESS
    )


def __calculate_dimensions(rectangle):
    # The width and height are the distances between the top
    # and left corner points, respectively.
    width = __euclidean(rectangle[0], rectangle[1])
    height = __euclidean(rectangle[0], rectangle[3])
    # Calculate the dimensions of the object inside the image,
    # then divide by the factor with which we upscale the image.
    return width / PAPER_SCALE, height / PAPER_SCALE


def __calculate_text_size(text):
    (width, height), baseline = cv2.getTextSize(
        text=text,
        fontFace=FONT,
        fontScale=FONT_SCALE,
        thickness=TEXT_THICKNESS
    )
    return width, height + baseline


def __euclidean(x, y):
    return np.linalg.norm(x - y)
