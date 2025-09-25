import cv2
import numpy as np
import os

ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}
def rotation_matrix_3d(axis, theta):
   
    c, s = np.cos(theta), np.sin(theta)

    if axis == 'x':
        return np.array([
            [1,  0,  0],
            [0,  c, -s],
            [0,  s,  c]
        ])
    elif axis == 'y':
        return np.array([
            [ c, 0,  s],
            [ 0, 1,  0],
            [-s, 0,  c]
        ])
    elif axis == 'z':
        return np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'.")

def make_directories(path):

    if not os.path.exists(path+"/JPEG"):
        os.makedirs(path+"/JPEG")
        print("create jpeg folder")




def drawTextToImage(txt,image):
    text = txt
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)  # White color
    font_thickness = 2

    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)

    image_height, image_width, _ = image.shape
    x = (image_width - text_width) // 2
    y = (image_height + text_height) // 2

    # Put the text on the image
    cv2.putText(image, text, (x, y), font, font_scale, font_color, font_thickness, cv2.LINE_AA)

def create_cube_board(cubeSize,markerSize,dict):
    marker_dis=0.008
    cube_dis=cubeSize/2
    marker_offset=marker_dis/2
    face1_marker1=np.array([[-(marker_offset+markerSize),-cube_dis,(marker_offset+markerSize)],
                            [-marker_offset             ,-cube_dis,(marker_offset+markerSize)],
                            [-marker_offset             ,-cube_dis,marker_offset],
                            [-(marker_offset+markerSize),-cube_dis,marker_offset]],dtype=np.float32)
    
    face1_marker2=np.array([[(marker_offset)            ,-cube_dis,(marker_offset+markerSize)],
                            [marker_offset+markerSize   ,-cube_dis,(marker_offset+markerSize)],
                            [marker_offset+markerSize   ,-cube_dis,marker_offset],
                            [marker_offset              ,-cube_dis,marker_offset]],dtype=np.float32)
    
    face1_marker3=np.array([[(marker_offset)            ,-cube_dis,-(marker_offset)],
                            [marker_offset+markerSize   ,-cube_dis,-(marker_offset)],
                            [marker_offset+markerSize   ,-cube_dis,-(marker_offset+markerSize)],
                            [marker_offset              ,-cube_dis,-(marker_offset+markerSize)]],dtype=np.float32)
    
    face1_marker4=np.array([[-(marker_offset+markerSize),-cube_dis,-(marker_offset)],
                            [-marker_offset             ,-cube_dis,-(marker_offset)],
                            [-marker_offset             ,-cube_dis,-(marker_offset+markerSize)],
                            [-(marker_offset+markerSize),-cube_dis,-(marker_offset+markerSize)]],dtype=np.float32)
    
    board_corners=[]
    board1=np.hstack((face1_marker1.T,face1_marker2.T,face1_marker3.T,face1_marker4.T))
    board_corners.append(face1_marker1)
    board_corners.append(face1_marker2)
    board_corners.append(face1_marker3)
    board_corners.append(face1_marker4)

    for i in range(1,6):
        rad=i*(-np.pi/2)
        if i < 4:
            rot=rotation_matrix_3d('z',rad)
        else:
            if i == 4:
                rad=-np.pi/2
            elif i == 5:
                rad=np.pi/2
            rot=rotation_matrix_3d('x',rad)
        next_board=np.matmul(rot,board1,dtype=np.float32).T
        next_board=next_board.reshape(4,4,3)
        board_corners.append(next_board[0])
        board_corners.append(next_board[1])
        board_corners.append(next_board[2])
        board_corners.append(next_board[3])
    board_ids=np.array([[0],[1],[8],[7],
                        [2],[3],[10],[9],
                        [5],[6],[13],[12],
                        [14],[15],[22],[21],
                        [17],[18],[25],[24],
                        [19],[20],[27],[26]],dtype=np.int32)
    board=cv2.aruco.Board(board_corners,dictionary=dict,ids=board_ids)

    return board

            
def create_grid_board(marker_size,aruco_dict,init_id=32): #door_init=0,box=init_10
    aruco_dict=cv2.aruco.getPredefinedDictionary(aruco_dict)
    aruco_params=cv2.aruco.DetectorParameters()
    nx=4
    ny=4
    gap_between_marker=0.1
    marker_ids=np.arange(init_id,init_id+(nx*ny))
    board = cv2.aruco.GridBoard((nx,ny),marker_size,gap_between_marker,aruco_dict,ids=marker_ids)
    
    out_size=(2000,2000)
    img_size=(1800,1800)
    img=board.generateImage(outSize=out_size,img=img_size,marginSize=20,borderBits=1)
    filename="AR_4x4_250_id{0}-{1}.jpg".format(init_id,init_id+(nx*ny)-1)
    cv2.imwrite(filename,img)
    return img




def main():
    img=create_grid_board(marker_size=0.8,aruco_dict=cv2.aruco.DICT_4X4_250)
    while(True):
        cv2.imshow("test",img)
        if cv2.waitKey(1) == ord('q'):
          break
    cv2.destroyAllWindows()


if __name__=="__main__":
    main()
