import time
import cv2
import numpy as np
import detection as dt
import ica

# STATES 
class State:
    IDLE = 0     # do nothing
    DETECT = 1   # find face & draw box
    MEASURE = 2  # collect data & calculate BPM

# FUNCTION FOR DISPLAYING MENU 
def display_menu(frame, lines):
    x0, y0 = 20, 50 # starting position
    dy = 40  # vertical spacing between lines
    for i, line in enumerate(lines):
        temp_y = y0 + i * dy
        # background rectangle for better visibility
        cv2.rectangle(frame, (x0 - 10, temp_y - 30), (x0 + 560, temp_y + 15), (0, 0, 0), -1)
        # actual text
        cv2.putText(frame, line, (x0, temp_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 200), 2)


def main():
    # webcam initialization
    cap = cv2.VideoCapture(0)

    # face detector and estimator initialization
    detector = dt.RegionDetector()
    
    # start in IDLE state
    current_state = State.IDLE

    # initialize selected region
    selected_region_id = None
    
    print("BPM ESTIMATOR - READY")
    print("Press 'q' to quit")

    # definition of menus for each state
    menu_idle = [
        "IDLE",
        " - Press 'd' to start face detection",
        " - Press 'q' to quit"
    ]
    
    menu_detect = [
        "DETECTING FACE...",
        " - Press 'n' to resturn to idle state",
        " - Press 'q' to quit",
        "SELECT THE CLEAREST REGION:",
        " - Press '1' for Left Cheek",
        " - Press '2' for Right Cheek",
        " - Press '3' for Forehead"
    ]
    
    menu_measure = [
        "MEASURING HEARTBEAT...",
        " - Press 'r' to restart bpm measurement",
        " - Press 'd' to restart Forehead detection",
        " - Press 'n' to resturn to idle state",
        " - Press 'q' to quit"
    ]

    # ICA Estimator
    ica_estimator = ica.Estimator(time.time())

    while True:
        ret, initial_frame = cap.read()
        if not ret: break

        frame = cv2.flip(initial_frame, 1)

        # KEYBOARD CONTROLS
        key = cv2.waitKey(1) & 0xFF

        if key == ord('n'): # NULL STATE
            current_state = State.IDLE
            print("State: IDLE")

        elif key == ord('d'): # DETECT STATE
            current_state = State.DETECT

            print("State: DETECT")

        elif key in [ord('1'), ord('2'), ord('3')]: # MEASURE STATE
            if current_state == State.MEASURE:
                print("Error: Already in MEASURE state!")             
            elif current_state == State.IDLE:
                print("Error: Must detect forehead first! Press 'd'.")
            else:
                selected_region_id = int(chr(key)) # convert key input to int
                current_state = State.MEASURE
                print(f"Selected Region ID: {selected_region_id}")
                print("State: MEASURE")
        
        elif key == ord('r'): # MEASURE STATE (RESTART)  
            if current_state != State.MEASURE:
                    print("Error: Region not selected! Press '1', '2', or '3' to select region.")
            else:
                # TODO: reset any variables and restart measurement
                print("Restarting measurement...")
                print("State: MEASURE")

        elif key == ord('q'): # QUIT
            # if ica_estimator.captures:
            #     for i, capture in enumerate(ica_estimator.captures):
            #         if i == 0:
            #             print(f"Frame {i}: {capture.time:.3f}s (first frame)")
            #         else:
            #             time_diff = capture.time - ica_estimator.captures[i-1].time
            #             print(f"Frame {i}: {capture.time:.3f}s (diff: {time_diff:.3f}s)")
            
            # ica_estimator.plot_channels()
            break

        # --- STATE MACHINE ---
        
        # IDLE STATE
        if current_state == State.IDLE:
            display_menu(frame, menu_idle)
        
        
        # DETECT STATE (runs in both DETECT and MEASURE states)
        elif current_state == State.DETECT or current_state == State.MEASURE:
            
            # display the correct menu
            if current_state == State.MEASURE:
                display_menu(frame, menu_measure)
            else:
                display_menu(frame, menu_detect)

            # run the face detection and forehead extraction
            detector.detect_face(frame)
            # optionally draw the face mesh
            # detector.draw_face_mesh(frame)

            # get region points for left cheek (1), right cheek (2), and forehead (3)
            left_cheek_points = detector.get_region_coords(frame, 1)
            right_cheek_points = detector.get_region_coords(frame, 2)
            forehead_points = detector.get_region_coords(frame, 3)

            if forehead_points is not None and left_cheek_points is not None and right_cheek_points is not None:
                # draw the forehead and left and right cheek polygons
                cv2.polylines(frame, [forehead_points], isClosed=True, color=(0, 0, 255), thickness=2)
                cv2.polylines(frame, [left_cheek_points], isClosed=True, color=(0, 0, 255), thickness=2)
                cv2.polylines(frame, [right_cheek_points], isClosed=True, color=(0, 0, 255), thickness=2)

                # MEASURE STATE 
                if current_state == State.MEASURE:  
                    # select the desired region 
                    if selected_region_id == 1:
                        region_points = left_cheek_points
                    elif selected_region_id == 2:
                        region_points = right_cheek_points
                    elif selected_region_id == 3: 
                        region_points = forehead_points
                    else:
                        region_points = None

                    cv2.polylines(frame, [region_points], isClosed=True, color=(0, 255, 0), thickness=2)

                    # extract roi signal means
                    roi_means = dt.extract_roi_means(frame, region_points)
                
                    if roi_means is not None:
                        mean_r, mean_g, mean_b = roi_means
                        # Print for debugging
                        # print(f"POLY:\n\tMean R: {mean_r}, Mean G: {mean_g}, Mean B: {mean_b}, Time: {time.time()}")

                        # add frame to ICA estimator
                        ica_estimator.add_frame(mean_r, mean_g, mean_b, time.time())

                        if ica_estimator.length() >= ica_estimator.capture_window/2 and ica_estimator.length() % 20 == 0:
                            print("Estimating BPM...")
                            start = time.time()
                            ica_estimator.estimate()
                            print(f"Estimation took {time.time() - start:.3f} seconds.")

                    # TODO: estimate BPM 

                    # cv2.putText(frame, bpm_display, (x, y), 
                    #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    pass
                
        # show the final frame
        cv2.imshow("BPM ESTIMATOR", frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()