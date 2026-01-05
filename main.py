import time
import cv2
import detection as dt
import estimation

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
        cv2.rectangle(frame, (x0 - 10, temp_y - 30), (x0 + 480, temp_y + 15), (0, 0, 0), -1)
        # actual text
        cv2.putText(frame, line, (x0, temp_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 200), 2)


def list_available_cameras(max_index=8):
    available = []
    for i in range(0, max_index + 1):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            cap.release()
            continue
        # try to grab a frame to be more certain the device works
        ret, _ = cap.read()
        if ret:
            available.append(i)
        cap.release()
    return available


def choose_camera():
    available = list_available_cameras(8)
    if not available:
        print("No cameras detected. Defaulting to index 0.")
        return 0
    if len(available) == 1:
        print(f"Found one camera at index {available[0]}. Using it.")
        return available[0]

    print("Available cameras:")
    for i in available:
        print(f"  [{i}] Camera {i}")

    while True:
        try:
            choice = input(f"Select camera index from {available} (press Enter to use {available[0]}): ")
        except KeyboardInterrupt:
            print("\nAborted by user. Exiting.")
            exit(0)

        if choice.strip() == "":
            return available[0]
        if choice.isdigit() and int(choice) in available:
            return int(choice)
        print("Invalid selection, try again.")


def main():
    # webcam initialization (choose from available cameras)
    cam_idx = choose_camera()
    cap = cv2.VideoCapture(cam_idx)

    try:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) 
        cap.set(cv2.CAP_PROP_EXPOSURE, -5.0) 
    except Exception as e:
        print("Could not set manual exposure:", e)

    # face detector and estimator initialization
    detector = dt.RegionDetector()
    
    # start in IDLE state
    current_state = State.IDLE
    
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
        " - Press 'n' to return to idle state",
        " - Press 'q' to quit",
        " - Press 'm' to start bpm measurement"
    ]
    
    menu_measure = [
        "MEASURING HEARTBEAT... STAY STILL!",
        " - Press 'r' to restart bpm measurement",
        " - Press 'd' to restart face detection",
        " - Press 'n' to return to idle state",
        " - Press 'q' to quit"
    ]

    # Estimator
    estimator = estimation.Estimator(time.time())
    last_bpm_display = "Measuring..."

    while True:
        ret, initial_frame = cap.read()
        if not ret: break

        frame = cv2.flip(initial_frame, 1)

        # KEYBOARD CONTROLS
        key = cv2.waitKey(1) & 0xFF

        if key == ord('n'): # NULL STATE
            current_state = State.IDLE
            last_bpm_display = "Measuring..."
            print("State: IDLE")

        elif key == ord('d'): # DETECT STATE
            current_state = State.DETECT
            last_bpm_display = "Measuring..."
            print("State: DETECT")

        elif key == ord('m'): # MEASURE STATE
            if current_state == State.MEASURE:
                print("Warning: Already in MEASURE state!")             
            elif current_state == State.IDLE:
                print("Error: Must detect face first! Press 'd' to detect face.")
            else:
                current_state = State.MEASURE
                estimator.captures = []  # reset captures
                estimator.estimations = [] # reset estimations history
                estimator.start_time = time.time()
                print("State: MEASURE")
        
        elif key == ord('r'): # MEASURE STATE (RESTART)  
            if current_state != State.MEASURE:
                    print("Error: Must detect face first! Press 'd' to detect face.")
            else:
                estimator.captures = []  # reset captures
                estimator.estimations = [] # reset estimations history
                estimator.start_time = time.time()
                last_bpm_display = "Measuring..."
                print("Restarting measurement...")
                print("State: MEASURE")

        elif key == ord('q'): # QUIT
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

            # get feature coordinates for the face
            features_coords = detector.get_all_feature_coords(frame)

            if features_coords is not None:
                # draw the forehead and left and right cheek polygons
                dt.draw_face_features(frame, features_coords)
                # MEASURE STATE 
                if current_state == State.MEASURE:  

                    cv2.polylines(frame, [features_coords["face_contour"]], isClosed=True, color=(0, 255, 0), thickness=2)
    
                    # extract roi signal means
                    roi_means = dt.extract_means(frame, features_coords)
                
                    if roi_means is not None:
                        mean_r, mean_g, mean_b = roi_means

                        # add frame to estimator
                        estimator.add_frame(mean_r, mean_g, mean_b, time.time())

                        if last_bpm_display.startswith("Measuring"):
                            last_bpm_display = f"Measuring... ({(estimator.length()/estimator.capture_window)*100:.0f}%)"
                        
                        if estimator.length() >= estimator.capture_window:
                            print("Estimating BPM...")
                            start = time.time()
                            bpm = estimator.estimate()
                            print(f"Estimation took {time.time() - start:.3f} seconds.")
                            if bpm is not None:
                                last_bpm_display = f"BPM: {bpm:.0f}"
                                print(f"Current Estimate: {bpm:.0f}\n")
                        
                        # Draw the BPM on screen over the top of the head
                        top_head_coords = detector.get_top_head_coords(frame)
                        if top_head_coords is not None:
                            x, y = top_head_coords
                            cv2.putText(frame, last_bpm_display, (x - 80, y - 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)                   
                
        # show the final frame
        cv2.imshow("BPM ESTIMATOR", frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()