import cv2
import mediapipe as mp
import numpy as np

class ForeheadDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        # indices for forehead region based on MediaPipe Face Mesh landmarks:
        # https://github.com/google-ai-edge/mediapipe/blob/e0eef9791ebb84825197b49e09132d3643564ee2/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
        self.FOREHEAD_INDICES_RECT = [109, 338, 108, 337] 
        # ploygon points for forehead region, more precise than rectangle
        self.FOREHEAD_INDICES = [109, 10, 338, 337, 151, 108] 
        self.landmark_results = None

    # Detect the face and get landmarks
    def detect_face(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.landmark_results = self.face_mesh.process(frame_rgb)
    
    # Optional: Draw the face mesh on the frame
    def draw_face_mesh(self, frame):
        if self.landmark_results.multi_face_landmarks:
            for face_landmarks in self.landmark_results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
    
    # get forehead coordinates as polygon points
    def get_forhead_coords(self, frame):
        if not self.landmark_results or not self.landmark_results.multi_face_landmarks:
            return None
        
        # get only the first detected face
        face_landmarks = self.landmark_results.multi_face_landmarks[0]
        height, width, _ = frame.shape

        points = []
        # extract specific coordinates for forehead indices
        for i in self.FOREHEAD_INDICES:
            x = int(face_landmarks.landmark[i].x * width)
            y = int(face_landmarks.landmark[i].y * height)
            points.append([x, y])

        return np.array(points)
    
def extract_roi_means(frame, points):
    # mask for the forehead region polygon
    # create an empty mask
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    # fill the polygon defined by points with white color (roi)
    cv2.fillPoly(mask, [points], 255)

    # calculate mean color within the masked region (only where pixels are white/255)
    # mean function returns (b, g, r, alpha)
    mean_b, mean_g, mean_r, _ = cv2.mean(frame, mask=mask)

    # return means as RGB 
    return (mean_r, mean_g, mean_b)
    
    ### RECTANGLE-BASED METHOD (easier but less efficient) ###

    # get forehead coordinates for rectangle (top-left and bottom-right points)
