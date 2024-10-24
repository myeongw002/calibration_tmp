#!/usr/bin/env python

import rospy
import threading
from sensor_msgs.msg import CompressedImage, PointCloud2
from std_msgs.msg import Float32
import sensor_msgs.point_cloud2 as pc2
from cv_bridge.boost.cv_bridge_boost import cvtColor2

import os
import cv2
import pcl
import numpy as np
import time
import matplotlib.pyplot as plt


class PointCloudProcessor:
    def __init__(self,  intrinsic_parameter):
        # Subscriber to the compressed image topic
        topic_list = [
            "/camera/image_color/compressed",
            "/ouster/points"
        ]
        upper_folder = "extrinsic_data"
        rospy.Subscriber(topic_list[0], CompressedImage, self.image_callback)
        rospy.Subscriber(topic_list[1], PointCloud2, self.falcon_callback)

        self.image_msg = CompressedImage()
        self.falcon_msg = PointCloud2()

        self.image_msg_list = []
        self.falcon_msg_list = []

        self.pub = rospy.Publisher("/data", Float32, queue_size=10)

        # Publisher for the pointcloud data
        # self.pointcloud_pub = rospy.Publisher("/ouster1/points", PointCloud2, queue_size=10)

        ### Folder list ###
        self.folder_list = []
        for topic_name in topic_list:
            folder_path = os.path.join(upper_folder,topic_name.split("/")[1])
            self.folder_list.append(folder_path)
            os.makedirs(folder_path, exist_ok=True)

        ### Intrinsic Parameter ###
        self.intrinsic = intrinsic_parameter[0]
        self.distortion = intrinsic_parameter[1]

        ### pattern parameter ###
        PATTERN_NUMBER = 5 # <--------------------------------
        self.objp = self.pattern_point_set()
        self.parameters, self.aruco_dict, self.pattern = self.charuco_parameter_set(PATTERN_NUMBER)

        ### image processing ###
        self.befor_corner = np.array([])
        self.befor_index = np.array([])
        self.befor_location = np.array([])
        self.befor_offset = 0.0
        self.offset = 0.0

        self.image = None
        self.chessboard_corners_c = None

        ### Data Save ###
        self.detect_mode = "translate" # <--------------------------------pixel, translate
        self.save_start = False
        self.save_end = False
        self.save_count = 0

        ### Thread ###
        self.lock = threading.Lock()
        # Start the thread for publishing pointcloud data
        self.publish_thread = threading.Thread(target=self.publish_loop)
        self.publish_thread.start()

    ######### 패턴판 정의 함수 ############

    def pattern_point_set(self, Checkerboard_size=0.170, Checkerboard_number=(4,5)):
        # 체커보드 크기 정의
        patten_size = Checkerboard_size # 0.095 # [m 미터]
        # 체커보드의 차원 정의
        CHECKERBOARD = Checkerboard_number # (7,5) # 체커보드 행과 열당 내부 코너 수
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # 각 체커보드 이미지에 대한 3D 점 벡터를 저장할 벡터 생성
        objpoints = []
        # 각 체커보드 이미지에 대한 2D 점 벡터를 저장할 벡터 생성
        imgpoints = [] 
        # 3D 점의 세계 좌표 정의
        objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
        objp = objp*patten_size

        objp = objp.reshape(-1).reshape((Checkerboard_number[0]*Checkerboard_number[1],3))
        return objp

    ### ArUco 패턴판 파라미터 ###
    def charuco_parameter_set(self, id=0, board_count=(4,5), size=170):
        ### ChArUco Parameter ###
        parameters = cv2.aruco.DetectorParameters_create()
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)

        # 가로 사각형 개수, 세로 사각형 개수, 체스보드 크기, 마커 크기, 마커 파라미터
        board_w_count = board_count[0] + 1
        board_h_count = board_count[1] + 1

        # 마커와 아루코 마커 사이의 크기비율
        board_Ch_length = 40
        board_Aru_length = 30

        # 실제 마커 사이즈
        marker_size = size

        # ID 범위
        id_range_count = id

        pattern = cv2.aruco.CharucoBoard_create(board_w_count, board_h_count, board_Ch_length, board_Aru_length, aruco_dict)
        pattern.ids += 15*id_range_count # 15, 30, 
        return parameters, aruco_dict, pattern

    #####################

    def charuco_detect(self, image):
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        marker_corners_c, marker_ids, rejectedImgPoints = cv2.aruco.detectMarkers(img_gray, self.aruco_dict, parameters=self.parameters)
        [retval, chessboard_corners_c, chessboard_ids_c] = \
                                    cv2.aruco.interpolateCornersCharuco(marker_corners_c, marker_ids, img_gray, self.pattern)
        index = chessboard_ids_c.reshape(-1)
        return index, chessboard_corners_c

    def pattern_translate(self, index, chessboard_corners_c, fisheye=False):
        ### 이미지 plane translate calculate ###
        pattern_point_3d = self.objp[index]
        translate = self.pattern_coordinate_calculate(pattern_point_3d, chessboard_corners_c, fisheye, self.intrinsic, self.distortion)
        return translate

    ### 카메라에 대한 패턴판 좌표 계산 ###
    def pattern_coordinate_calculate(self, objp, corners2, fisheye, intrinsic, distortion):
        # 패턴 좌표계 구하기
        figure_points_3D = objp
        image_points_2D = corners2
        if fisheye == 1:
            undistortion_image_points_2D = cv2.fisheye.undistortPoints(image_points_2D, intrinsic, distortion) 
            success, R, T = cv2.solvePnP(figure_points_3D, undistortion_image_points_2D, 
                                        intrinsic, distortion, flags=0)
        else :
            success, R, T = cv2.solvePnP(figure_points_3D, image_points_2D, 
                                        intrinsic, distortion, flags=0)

        R = cv2.Rodrigues(R)[0]
        translate = np.vstack( (np.hstack((R,T)), np.array([0,0,0,1]) ) )
        return translate





    def corner_display(self, image, chessboard_corners_c, color):
        for index in chessboard_corners_c:
            cv2.circle(image, np.array(index[0]).astype(int), 5, color, thickness=-1, lineType=None, shift=None)

        cv2.namedWindow('image', flags=cv2.WINDOW_NORMAL)
        cv2.imshow("image", image)
        cv2.waitKey(1)


    def image_callback(self, msg):
        # rospy.loginfo("Compressed image received")
        with self.lock:
            st = time.time()
            ### msg save ###
            self.image_msg = msg
            # image msg to array 
            self.image = self.CompressedImage_to_numpy(msg)
            # corner detect
            index, self.chessboard_corners_c = self.charuco_detect(self.image)
            corner = np.array(self.chessboard_corners_c).reshape(-1).reshape(len(index),2)
            
            if self.befor_corner.shape[0] != 0:
                if self.detect_mode == "pixel":
                    same_index = []
                    same_befor_index = []
                    for i, now_idx in enumerate(index):
                        for j, bf_idx in enumerate(self.befor_index):
                            if now_idx == bf_idx:
                                same_index.append(i)
                                same_befor_index.append(j)
                                break
                    same_corner = corner[same_index]
                    same_befor_corner = self.befor_corner[same_befor_index]
                    self.offset = np.sum(np.linalg.norm(same_corner - same_befor_corner,2,axis=1))
                elif self.detect_mode == "translate":
                    translate = self.pattern_translate(index, self.chessboard_corners_c, fisheye=False)
                    location = translate[:3,3]
                    if self.befor_location.shape[0] != 0:   
                        self.offset = np.linalg.norm(self.befor_location - location, 2)*1000
                        # color = (0, 0, 255)
                        # self.corner_display(self.image, self.chessboard_corners_c, color)
                ##########################
                ### reset pattern data ###
                self.befor_corner = corner
                self.befor_index = index
                self.befor_location = location
            else :
                self.befor_corner = corner
                self.befor_index = index
            # print(time.time()-st)

    def falcon_callback(self, msg):
        with self.lock:
            self.falcon_msg = msg


    ### msg to numpy ###
    def CompressedImage_to_numpy(self, msg):
        """
        from cv_bridge.boost.cv_bridge_boost import cvtColor2
        """
        # buf = np.frombuffer(msg.data, np.uint8)
        # image = cvtColor2(cv2.imdecode(buf, cv2.IMREAD_ANYCOLOR), 'bayer_rggb8', 'bgr8')

        buf = np.frombuffer(msg.data, np.uint8)
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)

        return image
    
    def PointCloud2_to_numpy(self, msg):
        """
        import sensor_msgs.point_cloud2 as pc2
        """
        pcd = list(pc2.read_points(msg, skip_nans=True, field_names=("x","y","z","intensity")))
        points=np.asanyarray(list(pcd))
        return points


    def data_append(self):
        self.image_msg_list.append(self.image_msg)
        self.falcon_msg_list.append(self.falcon_msg)

        self.save_count = len(self.image_msg_list)
        file_name = str(self.save_count).zfill(4)
        print(f"{file_name} Append !!!")

    def data_save(self, folder_list):
        for i in range(len(self.image_msg_list)):
            st = time.time()
            image = self.CompressedImage_to_numpy(self.image_msg_list[i])
            falcon_pcd = self.PointCloud2_to_numpy(self.falcon_msg_list[i])

            file_name = str(i+1).zfill(4)

            cv2.imwrite(os.path.join(self.folder_list[0], file_name+".jpg"), image)
            ### pcl 변환 ###
            pc = pcl.PointCloud_PointXYZI()
            pc.from_list(falcon_pcd[:,:4].tolist()) # XYZI
            pcl.save(pc, os.path.join(self.folder_list[1], file_name+".pcd"))

            print(f"{file_name} Save!!! {time.time()-st}")

    def publish_loop(self):
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            with self.lock:
                if self.offset != self.befor_offset:
                    differential = self.offset - self.befor_offset

                    data_msg = Float32()
                    data_msg.data = differential
                    self.pub.publish(data_msg)

                    # print(differential)
                if self.offset != 0:
                    color = (0, 0, 255)
                    ### 저장 타이밍 함수 ###
                    if self.save_start == True and self.save_end == False:
                        # 해당 위치에 함수 설정
                        # self.data_save(self.folder_list)
                        self.data_append()

                        color = (0, 255, 0)
                        self.save_end = True
                    ####################
                    if differential < 13 :
                        self.save_start = True
                    else :
                        self.save_end = False
                        self.save_start = False
                        color = (0, 0, 255)
                    self.corner_display(self.image, self.chessboard_corners_c, color)
            rate.sleep()
        print("SAVE Process Start")
        self.data_save(self.folder_list)
        print("SAVE Process End")

if __name__ == '__main__':
    rospy.init_node('pointcloud_processor_node', anonymous=True)

    # intrinsic = np.array([[848.018213, -0.875069, 970.050807],
    #                       [       0.0,849.002273, 612.744961],
    #                       [       0.0,       0.0,        1.0]])
    # distortion = np.array([-0.022594, 0.030614, -0.001038, -3.5E-05])

    intrinsic_param_file_dir = f"/home/myungw00/ROS/calibration/24252427" + "/intrinsic.csv"
    intrinsic_param = np.loadtxt(intrinsic_param_file_dir,delimiter=',',usecols=range(9))

    intrinsic = np.array([[intrinsic_param[0], intrinsic_param[1], intrinsic_param[2]],
                          [               0.0, intrinsic_param[3], intrinsic_param[4]],
                          [               0.0,                0.0,                1.0]])
    distortion = np.array(intrinsic_param[5:])

    intrinsic_parameter = (intrinsic, distortion)

    processor = PointCloudProcessor(intrinsic_parameter)
    rospy.spin()
