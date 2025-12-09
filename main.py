import cv2
import numpy as np

# STATES 
class State:
    IDLE = 0     # Do nothing
    DETECT = 1   # Find face & draw box
    MEASURE = 2  # Collect data & calculate BPM

# FUNCTION FOR DISPLAYING MENU 
def display_menu(frame, lines):
    x0, y0 = 20, 50 # starting position
    dy = 40  # vertical spacing between lines
    for i, line in enumerate(lines):
        temp_y = y0 + i * dy
        cv2.putText(frame, line, (x0, temp_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)


def main():
    # Webcam Initialization
    cap = cv2.VideoCapture(0)

    # Face Detector and Estimator Initialization ?
    
    
    # Start in IDLE state
    current_state = State.IDLE
    
    print("Controls:")
    print(" [d] - Start/Restart Face Detection")
    print(" [b] - Start/Restart BPM Estimation")
    print(" [n] - Return to idle state (NULL)")
    print(" [q] - Quit")

    # Definition of menus for each state
    menu_idle = [
        "IDLE",
        " - Press 'd' to start face detection",
        " - Press 'q' to quit"
    ]
    
    menu_detect = [
        "DETECTING FACE...",
        " - Press 'b' to start bpm measurement",
        " - Press 'd' to restart face detection",
        " - Press 'n' to resturn to idle state",
        " - Press 'q' to quit"
    ]
    
    menu_measure = [
        "MEASURING HEARTBEAT...",
        " - Press 'b' to restart bpm measurement",
        " - Press 'd' to restart face detection",
        " - Press 'n' to resturn to idle state",
        " - Press 'q' to quit"
    ]

    while True:
        ret, frame = cap.read()
        if not ret: break

        # KEYBOARD CONTROLS
        key = cv2.waitKey(1) & 0xFF

        if key == ord('n'): # NULL STATE
            current_state = State.IDLE
            print("State: IDLE")

        elif key == ord('d'): # DETECT STATE
            current_state = State.DETECT
            # processor = await HeartRateEstimator(buffer_size=150) 
            print("State: DETECT")

        elif key == ord('b'): # MEASURE STATE
            if current_state == State.IDLE:
                print("Error: Must detect face first! Press 'd'.")
            else:
                current_state = State.MEASURE
                # processor.reset_buffer() 
                print("State: MEASURE")

        elif key == ord('q'): # QUIT
            break

        # - STATE MACHINE -
        
        # IDLE STATE
        if current_state == State.IDLE:
            display_menu(frame, menu_idle)
        

        # DETECT STATE (runs in both DETECT and MEASURE states)
        elif current_state == State.DETECT or current_state == State.MEASURE:
            
            # Display the correct menu
            if current_state == State.MEASURE:
                display_menu(frame, menu_measure)
            else:
                display_menu(frame, menu_detect)

            # run the face detection ?

            
                # MEASURE STATE 
                if current_state == State.MEASURE:                    
                    # Estimate BPM ?

                    # cv2.putText(frame, bpm_display, (x, y-15), 
                    #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    pass
                
        # Show the final frame
        cv2.imshow("BPM ESTIMATOR", frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()