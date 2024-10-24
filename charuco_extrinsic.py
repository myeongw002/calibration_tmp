# pip install opencv-contrib-python==4.6.0.66
import cv2
import numpy as np
import os
import copy
import glob
import open3d as o3d
from scipy.optimize import least_squares
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

### Lidar Extrinsic Calibration ###
# https://stackoverflow.com/questions/74878923/open3d-registration-with-icp-shows-error-of-0-and-returns-the-input-transformati
def icp(source, target, trans_init):
    source.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    target.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # # Downsample the point clouds for better performance
    source = source.voxel_down_sample(voxel_size=0.05)
    target = target.voxel_down_sample(voxel_size=0.05)

    # TransformationEstimationPointToPoint
    # TransformationEstimationPointToPlane
    ### Point to Plane ###
    # 초기 변환 구하기
    # max_correspondence_distance : Nearest Neighbor Search를 위한 검색거리 [Hybrid-Search is used].
    # max_correspondence_distance 값이 보다 크면 icp가 작동안됨, 따라서 먼저 큰값으로 넣고 icp 진행
    max_correspondence_distance = 10
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    
    # 초기값 취득후 작은값으로 icp 진행
    max_correspondence_distance = 0.05
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance, reg_p2p.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())

    # print(reg_p2p)
    print("Transformation is:")
    print(trans_init)
    print(reg_p2p.transformation)
    print("------------------")
    return reg_p2p.transformation


def merge_point_clouds(pcd_list):
    if not pcd_list:
        raise ValueError("The point cloud list is empty")
    
    # Initialize a new empty PointCloud
    merged_pcd = o3d.geometry.PointCloud()

    for pcd in pcd_list:
        merged_pcd += pcd

    return merged_pcd

def icp_display(ouster1_pcd, ouster2_pcd, ouster3_pcd, os2_os1_translate, os2_os3_translate):

    ouster1_pcd = copy.deepcopy(ouster1_pcd).transform(os2_os1_translate)
    ouster3_pcd = copy.deepcopy(ouster3_pcd).transform(os2_os3_translate)

    ouster1_pcd.paint_uniform_color([1, 0, 0])
    ouster2_pcd.paint_uniform_color([0, 1, 0])
    ouster3_pcd.paint_uniform_color([0, 0, 1])
    # o3d.visualization.draw_geometries([ouster1_pcd, ouster2_pcd, ouster3_pcd])

    merged_pcd = merge_point_clouds([ouster1_pcd, ouster2_pcd, ouster3_pcd])
    theta = 10
    R = merged_pcd.get_rotation_matrix_from_xyz((0, theta*np.pi/180, 0))
    merged_pcd.rotate(R, center=(0, 0, 0))
    merged_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    mesh.scale(3, center=mesh.get_center())
    o3d.visualization.draw_geometries([mesh, merged_pcd])




### Pattern translate calulate ###
def pattern_point_set(Checkerboard_size=0.170, Checkerboard_number=(4,5)):
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

def charuco_parameter_set(id=0, board_count=(4,5), size=170):
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

def charuco_detect(image, parameters, aruco_dict, pattern):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    marker_corners_c, marker_ids, rejectedImgPoints = cv2.aruco.detectMarkers(img_gray, aruco_dict, parameters=parameters)
    [retval, chessboard_corners_c, chessboard_ids_c] = \
                                cv2.aruco.interpolateCornersCharuco(marker_corners_c, marker_ids, img_gray, pattern)
    index = chessboard_ids_c.reshape(-1)
    return index, chessboard_corners_c

### 카메라에 대한 패턴판 좌표 계산 ###
def pattern_coordinate_calculate(objp, corners, intrinsic, distortion, fisheye=False):
    # 패턴 좌표계 구하기
    figure_points_3D = objp
    image_points_2D = corners
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

### Plane fitting ###
def euler_trans2matrix(param):
    Deg2Rad = np.pi/180
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    T = np.eye(4)
    T[:3, :3] = mesh.get_rotation_matrix_from_xyz(np.array([param[0], param[1], param[2]])*Deg2Rad)
    T[0, 3] = param[3]
    T[1, 3] = param[4]
    T[2, 3] = param[5]
    return T

def pointcloud_range_filter(pcd, distance_threshold=30, query_point = [0, 0, 0]):
    # 거리 기준으로 필터링
    # query_point = [0, 0, 0]  # 거리를 측정할 기준점 좌표
    # distance_threshold = 100  # 거리 임계값 (미터)

    # 기준점과의 거리를 계산하여 거리 임계값 이내의 포인트만 남깁니다.
    points = np.asarray(pcd.points)
    distances = np.linalg.norm(points - query_point, axis=1)
    filtered_indices = np.where(distances <= distance_threshold)[0]

    # 거리 임계값 이내의 포인트 클라우드 생성
    filtered_pcd = pcd.select_by_index(filtered_indices)
    return filtered_pcd

def create_bounding_box(width, height, depth, offset=0.2):
    # 정육면체를 생성합니다.
    vertices = np.array([
        [0, 0, depth/2],
        [width, 0, depth/2],
        [width, height, depth/2],
        [0, height, depth/2],
        
        [0, 0, depth/-2],
        [width, 0, depth/-2],
        [width, height, depth/-2],
        [0, height, depth/-2]
    ])
    offset_array = np.array([
        [-offset, -offset, 0],
        [offset, -offset, 0],
        [offset, offset, 0],
        [-offset, offset, 0],

        [-offset, -offset, 0],
        [offset, -offset, 0],
        [offset, offset, 0],
        [-offset, offset, 0]
    ])
    vertices = vertices + offset_array

    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]

    # 정육면체의 점과 선을 Open3D의 LineSet으로 변환합니다.
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(vertices)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color([1, 0, 0])  # RGB 값 (빨간색)
    
    return line_set

def plane_point_roi(pcd_, init_trans, translate, pattern_size, Display=False):
    pcd = copy.deepcopy(pcd_)

    # pcd.transform(init_trans)
    # pcd.transform(translate)
    # pcd.transform(np.linalg.inv(init_trans))
    # pcd.transform(np.linalg.inv(init_trans) @ np.linalg.inv(translate))

    pcd.transform(np.linalg.inv(translate) @ np.linalg.inv(init_trans))
    pattern_coord_points = np.asarray(pcd.points)

    # 바운딩 박스 생성
    pattern_coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame()
    bounding_box = create_bounding_box(pattern_size[0], pattern_size[1], 0.5, offset=0.15)

    bbox_corners = np.asarray(bounding_box.points)
    min_bound = bbox_corners.min(axis=0)
    max_bound = bbox_corners.max(axis=0)

    # 바운딩 박스 경계 내에 있는 포인트를 필터링합니다.
    filtered_indices = np.where((pattern_coord_points[:, 0] >= min_bound[0]) & (pattern_coord_points[:, 0] <= max_bound[0]) &
                                (pattern_coord_points[:, 1] >= min_bound[1]) & (pattern_coord_points[:, 1] <= max_bound[1]) &
                                (pattern_coord_points[:, 2] >= min_bound[2]) & (pattern_coord_points[:, 2] <= max_bound[2]))[0]
    if Display == True:
        o3d.visualization.draw_geometries([pcd, pattern_coordinate, bounding_box])

    # ROI 패턴판
    plane_pts = pcd.select_by_index(filtered_indices)
    plane_pts_points = np.asarray(plane_pts.points)
    if plane_pts_points.shape[0] < 50:
        print("plane pointcloud number is small")
        return False, o3d.geometry.PointCloud(), bounding_box
    plane_model, inliers = plane_pts.segment_plane(distance_threshold=0.02,\
                                            ransac_n=3,\
                                            num_iterations=1000)
    
    pattern_point = o3d.geometry.PointCloud()
    pattern_point.points = o3d.utility.Vector3dVector(plane_pts_points[inliers])
    pattern_point.transform(init_trans @ translate)
    bounding_box.transform(init_trans @ translate)
    return True, pattern_point, bounding_box

def pointcloud_plane_list_extract(pcd_file_list, init_trans, pattern_translate_list, Checkerboard_number, pattern_size, Display=False, Reprocessing=True):
    pointcloud_plane_list = []
    for idx, file_path in enumerate(pcd_file_list):
        plane_pcd_folder_name = file_path.split("/")[-2]
        
        plane_plane_file_path = file_path.replace(plane_pcd_folder_name, plane_pcd_folder_name+"_plane")
        plane_plane_folder = "/".join( plane_plane_file_path.split("/")[:-1] )
        os.makedirs(plane_plane_folder, exist_ok=True)
        print(plane_plane_file_path)

        if Reprocessing == False:
            if os.path.exists(plane_plane_file_path) == True:
                roi_pattern_point = o3d.io.read_point_cloud(plane_plane_file_path)
                pointcloud_plane_list.append(roi_pattern_point)

                if Display == True:
                    lidar_coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame()
                    o3d.visualization.draw_geometries([lidar_coordinate, roi_pattern_point])
                continue

        # pattern coordinate
        pattern_translate = pattern_translate_list[idx]

        # pointcloud range filter
        pointcloud = pointcloud_range_filter( o3d.io.read_point_cloud(file_path), distance_threshold=20)

        # pattern point ROI
        success, roi_pattern_point, bounding_box = plane_point_roi(pointcloud, init_trans, pattern_translate, 
                                                                   pattern_size=np.array(Checkerboard_number)*pattern_size, Display=Display)
        o3d.io.write_point_cloud(plane_plane_file_path, roi_pattern_point)
        pointcloud_plane_list.append(roi_pattern_point)
    return pointcloud_plane_list


### Optimation ###
def plot_rmse(rmse_list, not_use_list):
    # RMSE 리스트의 개수를 N으로 설정
    N = len(rmse_list)
    
    # 1부터 N까지의 인덱스 생성
    indices = list(range(1, N+1))
    
    # 막대그래프 생성
    plt.figure(figsize=(10, 6))
    plt.bar(indices, rmse_list, color='skyblue')
    
    # 그래프 제목과 축 레이블 설정
    plt.title('RMSE Values Bar Graph')
    plt.xlabel('Index')
    plt.ylabel('RMSE')
    
    # x축과 y축 눈금 설정
    plt.xticks(indices)
    
    # 그래프 표시
    plt.show()

def cost_function(param, pattern_point_array, pattern_trans_array, not_use_list=[], rmse_display=False, rmse_plot_show=False):
    T = euler_trans2matrix(param)
    cost_array = []
    rmse_list = []
    for i in range(0,len(pattern_point_array)):
        if i+1 in not_use_list:
            rmse = 0
            rmse_list.append(rmse)
            continue
        # pattern_point = o3d.geometry.PointCloud()
        pattern_point = copy.deepcopy(pattern_point_array[i])
        # pattern_point.transform(T) # 카메라 좌표계
        # pattern_point.transform(np.linalg.inv(pattern_trans_array[i])) # 패턴 좌표계
        ### open3d 변환이용하면 이게 맞음 ###
        pattern_point.transform(np.linalg.inv(pattern_trans_array[i]) @ np.linalg.inv(T)) # 패턴 좌표계

        trans_pattern_point = np.asarray(pattern_point.points)

        cost = trans_pattern_point[:,2].tolist()
        cost_array = cost_array + cost

        rmse = np.sqrt(np.sum(np.power(cost,2))/len(cost))
        rmse_list.append(rmse)

    if rmse_display == True:
        rmse = np.sqrt(np.sum(np.power(cost_array,2))/len(cost_array))
        print(f"RMSE : {rmse}")
    if rmse_plot_show == True:
        plot_rmse(rmse_list, not_use_list)
    return cost_array


def sub_cost_function(T, pattern_point_array, pattern_trans_array, not_use_list=[], rmse_display=False, rmse_plot_show=False):
    # T = euler_trans2matrix(param)
    cost_array = []
    rmse_list = []
    for i in range(0,len(pattern_point_array)):
        if i+1 in not_use_list:
            rmse = 0
            rmse_list.append(rmse)
            continue
        # pattern_point = o3d.geometry.PointCloud()
        pattern_point = copy.deepcopy(pattern_point_array[i])
        # pattern_point.transform(T) # 카메라 좌표계
        # pattern_point.transform(np.linalg.inv(pattern_trans_array[i])) # 패턴 좌표계
        ### open3d 변환이용하면 이게 맞음 ###
        pattern_point.transform(np.linalg.inv(pattern_trans_array[i]) @ np.linalg.inv(T)) # 패턴 좌표계

        trans_pattern_point = np.asarray(pattern_point.points)

        cost = trans_pattern_point[:,2].tolist()
        cost_array = cost_array + cost

        rmse = np.sqrt(np.sum(np.power(cost,2))/len(cost))
        rmse_list.append(rmse)

    if rmse_display == True:
        rmse = np.sqrt(np.sum(np.power(cost_array,2))/len(cost_array))
        print(f"RMSE : {rmse}")
    if rmse_plot_show == True:
        plot_rmse(rmse_list, not_use_list)
    return cost_array


def cost_function_all(init_param, os2_os1_translate, os2_os3_translate, plane_list_all, pattern_translate_list):
    cost_list_all = []
    os2_cam_translate = euler_trans2matrix(init_param)
    for i in range(3):
        if i == 0 :
            extrinsic = np.linalg.inv(os2_os1_translate) @ os2_cam_translate
            cost_list = sub_cost_function(extrinsic, plane_list_all[i], pattern_translate_list, not_use_list=[], rmse_display=True)
            cost_list_all += cost_list
        elif i == 1 :
            extrinsic = os2_cam_translate
            cost_list = sub_cost_function(extrinsic, plane_list_all[i], pattern_translate_list, not_use_list=[], rmse_display=True)
            cost_list_all += cost_list
        elif i == 2 :
            extrinsic = np.linalg.inv(os2_os3_translate) @ os2_cam_translate
            cost_list = sub_cost_function(extrinsic, plane_list_all[i], pattern_translate_list, not_use_list=[], rmse_display=True)
            cost_list_all += cost_list
    print("---------")
    return cost_list_all



### Projection ###
def undistortion(x,y,distortion,r,fisheye=False):
    if fisheye == False and distortion.shape[0] == 5:
        X_Radial_dist = x*(1+distortion[0]*r**2+distortion[1]*r**4+distortion[4]*r**6)
        Y_Radial_dist = y*(1+distortion[0]*r**2+distortion[1]*r**4+distortion[4]*r**6)

        X_Tangential_dist = 2*distortion[2]*x*y+distortion[3]*(r**2+2*x**2)
        Y_Tangential_dist = distortion[2]*(r**2+2*y**2)+2*distortion[3]*x*y

        X_distortion = X_Radial_dist + X_Tangential_dist
        Y_distortion = Y_Radial_dist + Y_Tangential_dist

    elif fisheye == False and distortion.shape[0] == 4:
        X_Radial_dist = x*(1+distortion[0]*r**2+distortion[1]*r**4)
        Y_Radial_dist = y*(1+distortion[0]*r**2+distortion[1]*r**4)

        X_Tangential_dist = 2*distortion[2]*x*y+distortion[3]*(r**2+2*x**2)
        Y_Tangential_dist = distortion[2]*(r**2+2*y**2)+2*distortion[3]*x*y

        X_distortion = X_Radial_dist + X_Tangential_dist
        Y_distortion = Y_Radial_dist + Y_Tangential_dist

    elif fisheye == True:
        theat = np.arctan(r)
        theatD = theat*(1+distortion[0]*theat**2+distortion[1]*theat**4+\
                          distortion[2]*theat**6+distortion[3]*theat**8)
        X_distortion = (theatD/r)*x
        Y_distortion = (theatD/r)*y
    return X_distortion, Y_distortion

def projection(pointcloud, image_, extrinsic_list, intrinsic, distortion, fisheye, file_name, point_size=3, alpha=0.1, max_range=15, Save_file=True):
    projection_dir = "./projection/"
    os.makedirs(projection_dir, exist_ok=True)
    # matplot colormap
    cmap = plt.cm.get_cmap("jet", 256)                          # 256개의 색상 리스트 반환
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255 # 256개의 색상 리스트 반환
    # 포인트클라우드 값 추출
    lidar_pts = np.asarray(pointcloud.points)

    init_projection_result = None
    cal_projection_result = None
    for idx, extrinsic in enumerate(extrinsic_list):
        image =  copy.deepcopy(image_)

        length = np.linalg.norm(lidar_pts,2,axis=1)
        homo = extrinsic @ np.vstack(( lidar_pts.T, np.ones(lidar_pts.shape[0]) ))
        x = homo[0,:]/homo[2,:]; y = homo[1,:]/homo[2,:]
        r = np.sqrt(x**2 + y**2)
        X_distortion, Y_distortion = undistortion(x,y,distortion,r,fisheye=fisheye)
        pixel = intrinsic @ np.vstack(( X_distortion, Y_distortion, np.ones(X_distortion.shape[0]) ))

        # min_range = 3 # 디스토션 파라미터 핀홀 5개 사용 할때
        min_range = 0 # 디스토션 파라미터 핀홀 4개
        # max_range = 15
        index = np.where((pixel[0,:]>0) & (pixel[0,:]<image.shape[1]) & \
                        (pixel[1,:]>0) & (pixel[1,:]<image.shape[0]) & \
                        (homo[2,:] >min_range) )
        pixels = pixel.T[index,:2].astype(int)[0]
        color_index = (length[index]/max_range).astype(int)*255
        D = length[index]


        color_index = (255*D/max_range).astype(int)
        color_index[np.where(color_index > 255)] = 0
        color_map =  cmap[color_index]
        padding = 5
        image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        # 포인트 클라우드 센터
        image[padding+pixels[:,1],padding+pixels[:,0],:] = color_map

        pixel_range = np.arange(-point_size,point_size+1).tolist()
        for i in pixel_range:
            for j in pixel_range:
                image[padding+pixels[:,1]+i,padding+pixels[:,0]+j,:] = \
                    (image[padding+pixels[:,1]+i,padding+pixels[:,0]+j,:]*(1-alpha) + (color_map)*alpha ).astype(np.uint8)

        # 이미지의 네 가장자리에서 5픽셀을 제외한 부분 선택
        height, width, _ = image.shape
        projection_image = image[padding:height-padding, padding:width-padding]
        if idx == 0:
            init_projection_result = projection_image
        else :
            cal_projection_result = projection_image

    # cv2.imwrite("init_projection.png",init_projection_result)
    if Save_file == True:
        cv2.imwrite(projection_dir + file_name + ".png",cal_projection_result)
        #return 0
    
    display_image = cal_projection_result
    while True:
        cv2.namedWindow('image', flags=cv2.WINDOW_NORMAL)
        # cv2.resizeWindow(winname='image', width=1000, height=1000)
        cv2.imshow('image',display_image)
        key = cv2.waitKey(0)
        if key == ord("1"):
            display_image = image_
        elif key == ord("2"):
            display_image = init_projection_result
        elif key == ord("3"):
            display_image = cal_projection_result
        elif key == ord("q"):
            break


##########################################

def main():
    
    intrinsic_param_file_dir = f"/home/myungw00/ROS/calibration/24252427"
    intrinsic_param = np.loadtxt(intrinsic_param_file_dir + "/intrinsic.csv", delimiter=',',usecols=range(9))

    intrinsic = np.array([[intrinsic_param[0], intrinsic_param[1], intrinsic_param[2]],
                          [               0.0, intrinsic_param[3], intrinsic_param[4]],
                          [               0.0,                0.0,                1.0]])
    distortion = np.array(intrinsic_param[5:])
    print(intrinsic)
    print(distortion)
    print("\n")


    fisheye = False
    init_param = [-90, 90, 0, 0.0, 0.0, -0.0]

    # Pattern translate calulate
    PATTERN_NUMBER = 5
    Checkerboard_number=[4,5]
    pattern_size = 0.170 #mm


    ### Lidar Extrinsic Calibration ###
    print("\nLidar Extrinsic Calibration")

    ### Data path load ###
    print("Data path load")

    folder_name = "./extrinsic_data"
    image_file_list = sorted(glob.glob(folder_name+"/*/*.jpg", recursive=True))[:30]
    lidar_file_list_ = sorted(glob.glob(folder_name+"/*/*.pcd", recursive=True))[:30]
    lidar_file_list = []
    for lidar_file in lidar_file_list_:
        if "_plane" not in lidar_file:
            lidar_file_list.append(lidar_file)

    ### Pattern translate calulate ###
    print("\nPattern translate calulate")
    pattern_translate_list = []
    objp = pattern_point_set()
    parameters, aruco_dict, pattern = charuco_parameter_set(PATTERN_NUMBER)
    for image_file_path in image_file_list:
        image = cv2.imread(image_file_path)
        index, chessboard_corners_c = charuco_detect(image, parameters, aruco_dict, pattern)
        translate = pattern_coordinate_calculate(objp[index], chessboard_corners_c, intrinsic, distortion, fisheye=False)
        pattern_translate_list.append(translate)


    ### Plane fitting ###
    print("\nPlane fitting")
    init_trans = euler_trans2matrix(init_param)
    
    lidar_coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame()
    camera_coordinate = copy.deepcopy(lidar_coordinate)
    camera_coordinate.scale(0.5, center=(0, 0, 0))
    camera_coordinate.transform(init_trans)
    o3d.visualization.draw_geometries([lidar_coordinate, camera_coordinate])
    
    
    

    plane_list = pointcloud_plane_list_extract(lidar_file_list, init_trans, pattern_translate_list, 
                                                          Checkerboard_number, pattern_size, Display=False, Reprocessing=True)

    cost_function(init_param, plane_list, pattern_translate_list, rmse_display=True)
    print("==================== 최적화 결과 ======================")
    result = least_squares(cost_function, init_param, args=(plane_list, pattern_translate_list))
    result_translate_param = result.x
    print(result,'\n')
    print(result_translate_param.tolist(),'\n')
    cost_function(result_translate_param, plane_list, pattern_translate_list, rmse_display=True)
    lidar_cam_translate = euler_trans2matrix(result_translate_param)
    print("Extrinsic translate : ")
    print(np.linalg.inv(lidar_cam_translate))
    
    # Display
    for idx in range(len(lidar_file_list)):
        image = cv2.imread(image_file_list[idx])
        pcd = o3d.io.read_point_cloud(lidar_file_list[idx])
        file_name = image_file_list[idx].split('/')[-1].split('.')[0]
        projection(pcd, image, (np.linalg.inv(init_trans), np.linalg.inv(lidar_cam_translate)), 
                    intrinsic, distortion, fisheye, file_name, max_range=7, alpha=0.5, Save_file=True)


if __name__ == "__main__":
    main()
