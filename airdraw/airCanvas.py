import numpy as np
import cv2
from collections import deque


def set_values(x):
    pass


def initialize_trackbars():
    cv2.namedWindow("Color detectors")
    cv2.createTrackbar("Upper Hue", "Color detectors", 153, 180, set_values)
    cv2.createTrackbar("Upper Saturation", "Color detectors", 255, 255, set_values)
    cv2.createTrackbar("Upper Value", "Color detectors", 255, 255, set_values)
    cv2.createTrackbar("Lower Hue", "Color detectors", 64, 180, set_values)
    cv2.createTrackbar("Lower Saturation", "Color detectors", 72, 255, set_values)
    cv2.createTrackbar("Lower Value", "Color detectors", 49, 255, set_values)


def get_trackbar_values():
    u_hue = cv2.getTrackbarPos("Upper Hue", "Color detectors")
    u_saturation = cv2.getTrackbarPos("Upper Saturation", "Color detectors")
    u_value = cv2.getTrackbarPos("Upper Value", "Color detectors")
    l_hue = cv2.getTrackbarPos("Lower Hue", "Color detectors")
    l_saturation = cv2.getTrackbarPos("Lower Saturation", "Color detectors")
    l_value = cv2.getTrackbarPos("Lower Value", "Color detectors")
    return np.array([u_hue, u_saturation, u_value]), np.array([l_hue, l_saturation, l_value])


def create_paint_window():
    paint_window = np.zeros((471, 636, 3)) + 255
    paint_window = cv2.rectangle(paint_window, (40, 1), (140, 65), (0, 0, 0), 2)
    paint_window = cv2.rectangle(paint_window, (160, 1), (255, 65), (255, 0, 0), -1)
    paint_window = cv2.rectangle(paint_window, (275, 1), (370, 65), (0, 255, 0), -1)
    paint_window = cv2.rectangle(paint_window, (390, 1), (485, 65), (0, 0, 255), -1)
    paint_window = cv2.rectangle(paint_window, (505, 1), (600, 65), (0, 255, 255), -1)

    cv2.putText(paint_window, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(paint_window, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(paint_window, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(paint_window, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(paint_window, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 2, cv2.LINE_AA)

    return paint_window


def draw_buttons(frame):
    frame = cv2.rectangle(frame, (40, 1), (140, 65), (122, 122, 122), -1)
    frame = cv2.rectangle(frame, (160, 1), (255, 65), (255, 0, 0), -1)
    frame = cv2.rectangle(frame, (275, 1), (370, 65), (0, 255, 0), -1)
    frame = cv2.rectangle(frame, (390, 1), (485, 65), (0, 0, 255), -1)
    frame = cv2.rectangle(frame, (505, 1), (600, 65), (0, 255, 255), -1)
    cv2.putText(frame, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 2, cv2.LINE_AA)
    return frame


def main():
    initialize_trackbars()

    bpoints = [deque(maxlen=1024)]
    gpoints = [deque(maxlen=1024)]
    rpoints = [deque(maxlen=1024)]
    ypoints = [deque(maxlen=1024)]
    color_index = 0
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
    blue_index, green_index, red_index, yellow_index = 0, 0, 0, 0
    kernel = np.ones((5, 5), np.uint8)

    paint_window = create_paint_window()
    cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        upper_hsv, lower_hsv = get_trackbar_values()

        frame = draw_buttons(frame)

        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)

        cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center = None

        if cnts:
            cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
            ((x, y), radius) = cv2.minEnclosingCircle(cnt)
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            M = cv2.moments(cnt)
            center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

            if center[1] <= 65:
                if 40 <= center[0] <= 140:
                    bpoints = [deque(maxlen=1024)]
                    gpoints = [deque(maxlen=1024)]
                    rpoints = [deque(maxlen=1024)]
                    ypoints = [deque(maxlen=1024)]
                    blue_index, green_index, red_index, yellow_index = 0, 0, 0, 0
                    paint_window[67:, :, :] = 255
                elif 160 <= center[0] <= 255:
                    color_index = 0
                elif 275 <= center[0] <= 370:
                    color_index = 1
                elif 390 <= center[0] <= 485:
                    color_index = 2
                elif 505 <= center[0] <= 600:
                    color_index = 3
            else:
                if color_index == 0:
                    bpoints[blue_index].appendleft(center)
                elif color_index == 1:
                    gpoints[green_index].appendleft(center)
                elif color_index == 2:
                    rpoints[red_index].appendleft(center)
                elif color_index == 3:
                    ypoints[yellow_index].appendleft(center)
        else:
            if color_index == 0:
                bpoints.append(deque(maxlen=1024))
                blue_index += 1
            elif color_index == 1:
                gpoints.append(deque(maxlen=1024))
                green_index += 1
            elif color_index == 2:
                rpoints.append(deque(maxlen=1024))
                red_index += 1
            elif color_index == 3:
                ypoints.append(deque(maxlen=1024))
                yellow_index += 1

        points = [bpoints, gpoints, rpoints, ypoints]
        for i in range(len(points)):
            for j in range(len(points[i])):
                for k in range(1, len(points[i][j])):
                    if points[i][j][k - 1] is None or points[i][j][k] is None:
                        continue
                    cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                    cv2.line(paint_window, points[i][j][k - 1], points[i][j][k], colors[i], 2)

        cv2.imshow("Tracking", frame)
        cv2.imshow("Paint", paint_window)
        cv2.imshow("Mask", mask)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

'''Air-Canvas-project
Computer vision project implemented with OpenCV

Ever wanted to draw your imagination by just waiving your finger in air. In this post we will learn to build an Air Canvas which can draw anything on it by just capturing the motion of a coloured marker with camera. Here a coloured object at tip of finger is used as the marker.

We will be using the computer vision techniques of OpenCV to build this project. The preffered language is python due to its exhaustive libraries and easy to use syntax but understanding the basics it can be implemented in any OpenCV supported language.

Here Colour Detection and tracking is used in order to achieve the objective. The colour marker in detected and a mask is produced. It includes the further steps of morphological operations on the mask produced which are Erosion and Dilation. Erosion reduces the impurities present in the mask and dilation further restores the eroded main mask.

Algorithm
Start reading the frames and convert the captured frames to HSV colour space.(Easy for colour detection)
Prepare the canvas frame and put the respective ink buttons on it. 3.. Adjust the trackbar values for finding the mask of coloured marker.
Preprocess the mask with morphological operations.(Erotion and dilation)
Detect the contours, find the center coordinates of largest contour and keep storing them in the array for successive frames .(Arrays for drawing points on canvas)
Finally draw the points stored in array on the frames and canvas .
Requirements: python3 , numpy , opencv installed on your system.'''