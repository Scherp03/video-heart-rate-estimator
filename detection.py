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
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        # indices for forehead and cheek regions based on MediaPipe Face Mesh landmarks:
        # https://github.com/google-ai-edge/mediapipe/blob/e0eef9791ebb84825197b49e09132d3643564ee2/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
        # points manually selected to form polygons around the face and exclude eyes, eyebrows, and mouth
        self.FACE_INDECES = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 
                             379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 
                             234, 127, 162, 21, 54, 103, 67, 109
        ]
        self.LEFT_EYE_INDICES = [130, 247, 30, 29, 27, 28, 56, 190, 243, 112, 26, 22, 23, 24, 110, 25]
        self.RIGHT_EYE_INDICES = [359, 467, 260, 259, 257, 258, 286, 414, 463, 341, 256, 252, 253, 254, 339, 255]
        self.LEFT_EYEBROW_INDICES = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
        self.RIGHT_EYEBROW_INDICES = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
        self.MOUTH_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
        
        self.landmark_results = None

    # detect the face and get landmarks
    def detect_face(self, frame):
        # convert the BGR image to RGB before processing
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

    # function to convert list of indices to list of (x,y) points coords
    def _get_coords_from_indices(self, frame, indices):
        if not self.landmark_results or not self.landmark_results.multi_face_landmarks:
            return None
        
        # select the first detected face
        face_landmarks = self.landmark_results.multi_face_landmarks[0]
        height, width, _ = frame.shape
        points = []

        # convert normalized landmarks to pixel coordinates
        for i in indices:
            pt = face_landmarks.landmark[i]
            x = int(pt.x * width)
            y = int(pt.y * height)
            points.append([x, y])

        return np.array(points)

    # get all feature coordinates as a dictionary to facilitate manipulation and extraction
    def get_all_feature_coords(self, frame):
        return {
            "face_contour": self._get_coords_from_indices(frame, self.FACE_INDECES),
            "left_eye": self._get_coords_from_indices(frame, self.LEFT_EYE_INDICES),
            "right_eye": self._get_coords_from_indices(frame, self.RIGHT_EYE_INDICES),
            "left_eyebrow": self._get_coords_from_indices(frame, self.LEFT_EYEBROW_INDICES),
            "right_eyebrow": self._get_coords_from_indices(frame, self.RIGHT_EYEBROW_INDICES),
            "mouth": self._get_coords_from_indices(frame, self.MOUTH_INDICES)
        }
    
    # get the coordinates of the top's middle of the head (used for displaying text)
    def get_top_head_coords(self, frame):
        if not self.landmark_results or not self.landmark_results.multi_face_landmarks:
            return None
        
        # select the first detected face
        face_landmarks = self.landmark_results.multi_face_landmarks[0]
        height, width, _ = frame.shape

        # index for the top's middle of the forehead
        index = 10
    
        x = int(face_landmarks.landmark[index].x * width)
        y = int(face_landmarks.landmark[index].y * height)

        return (x, y)
    
# spatial averaging to extract mean RGB values from the face region
# create a binary mask for the face region excluding eyes, eyebrows, and mouth
# face is white (255), excluded features are black (0)
def extract_means(frame, features):
    # get frame dimensions
    height, width = frame.shape[:2]
    
    # create an empty/black mask
    mask = np.zeros((height, width), dtype=np.uint8)

    # draw the full face in white
    if features["face_contour"] is not None:
        cv2.fillPoly(mask, [features["face_contour"]], 255)

    # draw features in black to "remove" them
    exclusion_keys = ["left_eye", "right_eye", "left_eyebrow", "right_eyebrow", "mouth"]
    for key in exclusion_keys:
        points = features[key]
        if points is not None:
            cv2.fillPoly(mask, [points], 0)

    # perform spatial averaging within the masked region (only where pixels are white/255)
    # mean function returns (b, g, r, alpha)
    mean_b, mean_g, mean_r, _ = cv2.mean(frame, mask=mask)

    # return means as RGB 
    return (mean_r, mean_g, mean_b)

# draw the detected features on the frame for visualization purposes
def draw_face_features(frame, features): 
    # draw the full face contour in green
    if features["face_contour"] is not None:
        cv2.polylines(frame, [features["face_contour"]], isClosed=True, color=(0, 255, 0), thickness=2)
        # optional: fill face in white
        # cv2.fillPoly(frame, [features["face_contour"]], (255, 255, 255))

    # draw features that will be removed in black 
    for key in ["left_eye", "right_eye", "left_eyebrow", "right_eyebrow", "mouth"]:
        points = features[key]
        if points is not None:
            # draw contours in red
            cv2.polylines(frame, [points], isClosed=True, color=(0, 0, 255), thickness=2)
            # optional: fill features in black
            # cv2.fillPoly(frame, [points], (0, 0, 0))
