# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 20:43:58 2024

@author: homap
"""

#TASK 4

# Load the video
video_path = 'sample4.mov'  # Update with the correct path if needed
cap = cv2.VideoCapture(video_path)

# Define the Region of Interest (ROI) based on analysis
roi_start_x, roi_start_y = 120, 120  # Starting point of ROI
roi_width, roi_height = 240, 120  # Width and height of ROI

# Define motion detection thresholds and entry/exit zones
motion_threshold = 1500  # Minimum area of motion to consider significant
entry_threshold_y = roi_height * 0.6  # 60% of ROI height from the top for entry detection
exit_threshold_y = roi_height * 0.3  # 40% of ROI height from the top for exit detection

# Initialize counters and previous frame for motion detection
enter_count = 0
exit_count = 0
previous_frame = None

# Process each frame in the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale and isolate the ROI
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi_frame = gray_frame[roi_start_y:roi_start_y + roi_height, roi_start_x:roi_start_x + roi_width]

    # Initialize the previous frame if it hasn't been set yet
    if previous_frame is None:
        previous_frame = roi_frame
        continue

    # Calculate the difference between the current frame and the previous frame
    frame_diff = cv2.absdiff(previous_frame, roi_frame)
    _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
    previous_frame = roi_frame

    # Detect contours to find motion
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Process each contour to count entries and exits
    for contour in contours:
        if cv2.contourArea(contour) > motion_threshold:  # Only consider large motions
            x, y, w, h = cv2.boundingRect(contour)
            motion_center_y = y + h // 2  # Get the y-center of the motion

            # Count as entry if moving downward within the entry zone
            if motion_center_y > entry_threshold_y:
                enter_count += 1
            # Count as exit if moving upward within the exit zone
            elif motion_center_y < exit_threshold_y:
                exit_count += 1

# Release the video capture
cap.release()

# Display the final counts
print("Total Entries:", enter_count)
print("Total Exits:", exit_count)