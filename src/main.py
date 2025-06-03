import cv2
import numpy as np
import mediapipe as mp

mestre_landmarks = (
    149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 367, 288, 435, 433, 361, 401, 323, 366, 454, 447, 356,
    389, 251, 284, 332, 297, 338, 10, 109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 58, 172, 136, 150,10,152,345,116
    )
    
marionete_landmarks = (
    149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 367, 288, 435, 433, 361, 401, 323, 366, 454, 447, 356,
    389, 251, 284, 332, 297, 338, 10, 109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 58, 172, 136, 150,10,152,345,116
    )
    
face_triangles = (
        [17,176,400],
        [136,181,176],
        [181,17,176],
        [17,405,400],
        [405,400,365],
        [136,58,146],
        [146,181,136],
        [405,375,365],
        [375,435,365],
        [58,146,61],
        [375,435,409],
        [409,427,435],
        [427,323,435],
        [266,323,427],
        [266,427,409],
        [266,448,323],
        [448,447,323],
        [448,389,447],
        [58,61,132],
        [132,61,205],
        [132,205,234],
        [119,205,234],
        [234,127,119],
        [119,205,48],
        [165,205,48],
        [165,48,97],
        [165,164,97],
        [164,97,326],
        [326,164,391],
        [326,391,266],
        [165,391,0],
        [165,0,61],
        [0,391,409],
        [0,82,0],
        [61,82,0],
        [0,312,409],
        [61,178,181],
        [181,178,17],
        [178,17,317],
        [317,405,375],
        [97,326,4],
        [48,4,97],
        [4,326,266],
        [48,188,119],
        [188,4,350],
        [350,266,4],
        [350,448,266],
        [188,350,8],
        [188,55,8],
        [8,285,350],
        [107,55,8],
        [8,286,336],
        [107,8,336],
        [107,108,336],
        [108,336,337],
        [108,10,337],
        [10,337,338],
        [337,338,332],
        [10,109,108],
        [103,109,108],
        [103,105,108],
        [105,107,108],
        [332,334,337],
        [336,334,337],
        [105,21,103],
        [21,53,105],
        [53,105,107],
        [53,55,107],
        [127,33,119],
        [119,153,33],
        [153,119,188],
        [153,173,188],
        [173,55,188],
        [160,33,53],
        [160,157,55],
        [53,160,157],
        [157,173,55],
        [55,53,157],
        [350,380,448],
        [285,384,283],
        [285,334,283],
        [336,334,285],
        [387,466,283],
        [387,384,283],
        [380,390,448],
        [390,466,389],
        [390,448,389],
        [283,251,389],
        [283,389,466],
        [332,251,283],
        [283,334,332],
        [21,33,53],
        [127,21,33],
        [81,0,312],
        [391,165,0],
        [391,266,409],
        [61,205,165],
        [17,317,405],
        [188,48,4],
        [55,188,173],
        [178,82,61],
        [178,317,82],
        [317,82,312],
        [312,317,375],
        [312,375,409],
        [285,384,350],
        [380,384,387],
        [380,390,387],
        [387,390,263],
        [350,384,380],
        [33,160,153],
        [153,160,157],
        [153,173,157]
    )

# Function to draw circles around face landmarks
def draw_landmarks(image, landmarks, color):
    for landmark in landmarks.landmark:
        height, width, _ = image.shape
        cx, cy = int(landmark.x * width), int(landmark.y * height)
        #cv2.circle(image, (cx, cy), 1, color, -1) 

def calculate_homography(src_points, dst_points):
    src_points = np.array(src_points)
    dst_points = np.array(dst_points)
    homography_matrix, _ = cv2.findHomography(src_points, dst_points)
    return homography_matrix

def apply_homography(points, homography_matrix):
    points = np.array([points], dtype=np.float32)
    transformed_points = cv2.perspectiveTransform(points, homography_matrix)
    return transformed_points[0]

def getPoints(frame,result_processamento_mesh,landmarks):
    points = []
    
    if result_processamento_mesh.multi_face_landmarks:
        for landmarks2 in result_processamento_mesh.multi_face_landmarks:
            draw_landmarks(frame, landmarks2, (0, 0, 255))

            for idx in landmarks:
                height, width, _ = frame.shape
                cx, cy = int(landmarks2.landmark[idx].x * width), int(landmarks2.landmark[idx].y * height)
                points.append((cx, cy))
    
    return points

# Main function
def main():
    # Initialize mediapipe FaceMesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()
    face_mesh2 = mp_face_mesh.FaceMesh()

    # Inicializando Webcam do Mestre
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    #Inicializando Webcam da Marionete
    cap2 = cv2.VideoCapture(1)
    cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while cap.isOpened() and cap2.isOpened():
        ret, frame_master = cap.read()
        ret2, frame_puppet = cap2.read()
        
        if not ret and not ret2:
            break
        
        result = frame_puppet.copy()
        triangulation = []
        
        initial_master_points = []
        initial_puppet_points = []

        all_master_points = []

        # Convert BGR image to RGB
        rgb_frame = cv2.cvtColor(frame_master, cv2.COLOR_BGR2RGB)
        rgb_frame2 = cv2.cvtColor(frame_puppet, cv2.COLOR_BGR2RGB)

        # Process the frame and get face landmarks
        results_master = face_mesh.process(rgb_frame)
        results_puppet = face_mesh2.process(rgb_frame2)

        # Draw circles(Optional) around face landmarks if landmarks are detected
        initial_master_points = getPoints(frame_master,results_master,mestre_landmarks)
        initial_puppet_points = getPoints(frame_puppet,results_puppet,marionete_landmarks)        

        if len(initial_master_points) == 46 and len(initial_puppet_points) == 46:
            homography_matrix = calculate_homography(initial_master_points, initial_puppet_points)
            
            all_landmarks = np.arange(mp_face_mesh.FACEMESH_NUM_LANDMARKS)
            all_master_points = getPoints(frame_master,results_master,all_landmarks)
            all_puppet_points = getPoints(frame_puppet,results_puppet,all_landmarks)          

            if all_master_points is not None and all_puppet_points is not None:
                ''' 
                Pega os pontos do mestre, e mapeia-os para o plano da marionete.
                Ou seja, a ação/atitude do mestre é convertida para a marionete
                '''

                master2puppet_points = apply_homography(all_master_points, homography_matrix)
                
                #Nossa triangulação está feita no face_triangles
                
                for tripla in face_triangles:
                    #Coordenadas(x,y) do pontos do triângulo no plano da marionete
                    triangle_mestre_mapeado = np.array([master2puppet_points[x] for x in tripla])
                    triangle_puppet = np.array([all_puppet_points[x] for x in tripla])
                
                    matrix = cv2.getAffineTransform(triangle_puppet.astype(np.float32), triangle_mestre_mapeado.astype(np.float32))
                    warped_triangle = cv2.warpAffine(frame_puppet, matrix, (frame_puppet.shape[1], frame_puppet.shape[0]), borderMode=cv2.BORDER_TRANSPARENT,borderValue=0)
                    
                    mask = np.zeros_like(frame_puppet)
                    cv2.fillConvexPoly(mask, triangle_puppet, (255, 255, 255), cv2.LINE_4, 0)
                
                    #Adicionando o resultado da operação com o triângulo mestre_mapeado e o puppet na imagem final(result)
                
                    mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
                    inverse_mask = cv2.bitwise_not(mask_gray)
                    region_to_copy = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_gray)
                    region_to_replace = cv2.bitwise_and(result, result, mask=inverse_mask)
                    
                    #O resultado das operações são salvos aqui
                    result = cv2.add(region_to_copy, region_to_replace)

        # Display the output frame
        cv2.imshow("Master", frame_master)
        cv2.imshow("Puppet", frame_puppet)
        cv2.imshow("Result",result)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object
    cap.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
