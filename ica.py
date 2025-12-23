class Capture:
    def __init__(self, r = None, g = None, b = None, time = None):
        self.red = r
        self.green = g
        self.blue = b
        self.time = time
        
        

class Estimator:
    def __init__(self, start_time = None):
        self.capture_window = 300  # number of frames to consider
        self.captures = []
        self.start_time = start_time

    def add_frame(self, r, g, b, time):
        if r is not None and g is not None and b is not None:
            if len(self.captures) >= self.capture_window:
                print("Removing oldest capture")
                self.captures.pop(0)  # remove oldest capture
            
            self.captures.append(Capture(r, g, b, time))
            print(f"Added capture: R={r}, G={g}, B={b}, Time={time - self.start_time if self.start_time else time}")

        
        
    def length(self):
        print(f"Length of captures: {len(self.captures)}")
        return len(self.captures)

    def estimate(self):
        # Placeholder for ICA-based heart rate estimation logic
        pass