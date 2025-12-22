import cv2
import mediapipe as mp
import numpy as np

class RegionDetector:
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
        # indices for forehead and cheek regions based on MediaPipe Face Mesh landmarks:
        # https://github.com/google-ai-edge/mediapipe/blob/e0eef9791ebb84825197b49e09132d3643564ee2/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
        # ploygon points for forehead region, more precise than rectangle
        self.FOREHEAD_INDICES = [109, 10, 338, 337, 151, 108] 
        self.LEFT_CHEEK_INDICES = [123, 50, 205, 206, 216, 212, 214, 192, 213, 147]
        self.RIGHT_CHEEK_INDICES = [352, 280, 425, 426, 436, 432, 434, 416, 433, 376]
        self.landmark_results = None

    # detect the face and get landmarks
    def detect_face(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.landmark_results = self.face_mesh.process(frame_rgb)
    
    # optional: draw the face mesh on the frame
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

    def get_region_coords(self, frame, region):
        if region < 1 or region > 3:
            return None
        if not self.landmark_results or not self.landmark_results.multi_face_landmarks:
            return None
        
        # get only the first detected face
        face_landmarks = self.landmark_results.multi_face_landmarks[0]
        height, width, _ = frame.shape

        points = []

        if region == 1:
            indices = self.LEFT_CHEEK_INDICES
        elif region == 2:
            indices = self.RIGHT_CHEEK_INDICES
        else: # i.e. region == 3
            indices = self.FOREHEAD_INDICES

        # extract specific coordinates for given indices
        for i in indices:
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

    # calculate mean colors within the masked region (only where pixels are white/255)
    # mean function returns (b, g, r, alpha)
    mean_b, mean_g, mean_r, _ = cv2.mean(frame, mask=mask)

    # return means as RGB 
    return (mean_r, mean_g, mean_b)
