# Author: Minghao Gou

import numpy as np
import cv2
from numpy.core.numeric import full
import rospy
import rospkg
import open3d as o3d
import open3d_plus as o3dp
from transforms3d.quaternions import mat2quat
import time
import copy
import os
from copy import deepcopy

from .arm_controller import ArmController
from .pose.pose_6d import get_6d_pose_by_geometry, load_model_pcd
from .pose.pose_correspondence import get_pose_superglue

from ocrtoc_common.camera_interface import CameraInterface
from ocrtoc_common.transform_interface import TransformInterface
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
import pickle

class Perceptor2():
    def __init__(
        self,
        config
    ):
        self.config = config
        self.debug = self.config['debug']
        self.arm_controller = ArmController(topic = self.config['arm_topic'])
        rospy.sleep(2)
        self.transform_interface = TransformInterface()
        self.transform_from_frame = self.config['transform_from_frame']
        self.fixed_arm_poses = np.loadtxt(
            os.path.join(
                rospkg.RosPack().get_path('ocrtoc_perception'),
                self.config['realsense_camera_fix_pose_file'],
            ),
            delimiter = ','
        )
        self.fixed_arm_poses_both = np.loadtxt(
            os.path.join(
                rospkg.RosPack().get_path('ocrtoc_perception'),
                self.config['both_camera_fix_pose_file'],
            ),
            delimiter = ','
        )

    def assign_grasp_pose(self, gg, object_poses):
        grasp_poses = dict()
        dist_thresh = self.config['response']['dist_thresh']
        # - dist_thresh: float of the minimum distance from the grasp pose center to the object center. The unit is millimeter.
        angle_thresh = self.config['response']['angle_thresh']
        # - angle_thresh:
        #             /|
        #            / |
        #           /--|
        #          /   |
        #         /    |
        # Angle should be smaller than this angle

        object_names = []
        # gg: GraspGroup in 'world' frame of 'graspnet' gripper frame.
        # x is the approaching direction.
        ts = gg.translations
        rs = gg.rotation_matrices
        depths = gg.depths
        scores = gg.scores

        # move the center to the eelink frame
        ts = ts + rs[:,:,0] * (np.vstack((depths, depths, depths)).T)
        eelink_rs = np.zeros(shape = (len(rs), 3, 3), dtype = np.float32)

        # the coordinate systems are different in graspnet and ocrtoc
        eelink_rs[:,:,0] = rs[:,:,2]
        eelink_rs[:,:,1] = -rs[:,:,1]
        eelink_rs[:,:,2] = rs[:,:,0]

        # min_dist: np.array of the minimum distance to any object(must > dist_thresh)
        min_dists = np.inf * np.ones((len(rs)))

        # min_object_ids: np.array of the id of the nearest object.
        min_object_ids = -1 * np.ones(shape = (len(rs)), dtype = np.int32)


        # first round to find the object that each grasp belongs to.

        # Pay attention that even the grasp pose may be accurate,
        # poor 6dpose estimation result may lead to bad grasping result
        # as this step depends on the 6d pose estimation result.
        angle_mask = (rs[:, 2, 0] < -np.cos(angle_thresh / 180.0 * np.pi))
        for i, object_name in enumerate(object_poses.keys()):
            object_names.append(object_name)
            object_pose = object_poses[object_name]

            dists = np.linalg.norm(ts - object_pose['pose'][:3,3], axis=1)
            object_mask = np.logical_and(dists < min_dists, dists < dist_thresh)

            min_object_ids[object_mask] = i
            min_dists[object_mask] = dists[object_mask]
        remain_gg = []
        # second round to calculate the parameters
        for i, object_name in enumerate(object_poses.keys()):
            object_pose = object_poses[object_name]

            obj_id_mask = (min_object_ids == i)
            add_angle_mask = (obj_id_mask & angle_mask)
            # For safety and planning difficulty reason, grasp pose with small angle with gravity direction will be accept.
            # if no grasp pose is available within the safe cone. grasp pose with the smallest angle will be used without
            # considering the angle.
            if np.sum(add_angle_mask) < self.config['response']['mask_thresh']: # actually this should be mask == 0, for safety reason, < 0.5 is used.
                mask = obj_id_mask
                sorting_method = 'angle'
            else:
                mask = add_angle_mask
                sorting_method = 'score'
            if self.debug:
                print(f'{object_name} using sorting method{sorting_method}, mask num:{np.sum(mask)}')
            i_scores = scores[mask]
            i_ts = ts[mask]
            i_eelink_rs = eelink_rs[mask]
            i_rs = rs[mask]
            i_gg = gg[mask]
            if np.sum(mask) < self.config['response']['mask_thresh']: # actually this should be mask == 0, for safety reason, < 0.5 is used.
                # ungraspable
                grasp_poses[object_name] = None
            else:
                if sorting_method == 'score':
                    best_grasp_id = np.argmax(i_scores)
                elif sorting_method == 'angle':
                    best_grasp_id = np.argmin(i_rs[:, 2, 0])
                else:
                    raise ValueError('Unknown sorting method')
                best_g = i_gg[int(best_grasp_id)]
                remain_gg.append(best_g.to_open3d_geometry())
                grasp_rotation_matrix = i_eelink_rs[best_grasp_id]
                if np.linalg.norm(np.cross(grasp_rotation_matrix[:,0], grasp_rotation_matrix[:,1]) - grasp_rotation_matrix[:,2]) > 0.1:
                    if self.debug:
                        print('\033[031mLeft Hand Coordinate System Grasp!\033[0m')
                    grasp_rotation_matrix[:,0] = - grasp_rotation_matrix[:, 0]
                else:
                    if self.debug:
                        print('\033[032mRight Hand Coordinate System Grasp!\033[0m')
                gqw, gqx, gqy, gqz = mat2quat(grasp_rotation_matrix)
                grasp_poses[object_name] = {
                    'x': i_ts[best_grasp_id][0],
                    'y': i_ts[best_grasp_id][1],
                    'z': i_ts[best_grasp_id][2],
                    'qw': gqw,
                    'qx': gqx,
                    'qy': gqy,
                    'qz': gqz
                }
        return grasp_poses, remain_gg

    def percept(
            self,
            object_list,
            pose_method,
        ):
        '''
        Generate object 6d poses and grasping poses.
        Only geometry infomation is used in this implementation.

        There are mainly three steps.
        - Moving the camera to different predefined locations and capture RGBD images. Reconstruct the 3D scene.
        - Generating objects 6d poses by mainly icp matching.
        - Generating grasping poses by graspnet-baseline.

        Args:
            object_list(list): strings of object names.
            pose_method: string of the 6d pose estimation method, "icp" or "superglue".
        Returns:
            dict, dict: object 6d poses and grasp poses.
        '''
        # Capture Data
        arm_poses = np.array(self.fixed_arm_poses_both).tolist()[-1:]
        for arm_pose in arm_poses:
            self.arm_controller.exec_joint_goal(arm_pose)
            rospy.sleep(2.0)
            time.sleep(1.0)
        rospack = rospkg.RosPack()
        taskid = rospy.get_param('/pybullet_env/task_index')
        save_path = os.path.join(rospack.get_path('ocrtoc_perception'),'data', taskid,'')
        save_path = os.path.join(rospack.get_path('ocrtoc_perception'),'data','')
        full_pcd = o3d.io.read_point_cloud(save_path + 'full_pcd.pcd') 
        with open(save_path +  'color_images.pickle', 'rb') as handle:
            color_images = pickle.load(handle)  
        with open(save_path + 'camera_poses.pickle', 'rb') as handle:
            camera_poses = pickle.load(handle) 
        # Compute Grasping Poses (Many Poses in a Scene)
        with open(save_path + 'gg.pickle', 'rb') as handle:
            gg = pickle.load(handle)      
        if self.debug:
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
            o3d.visualization.draw_geometries([frame, full_pcd, *gg.to_open3d_geometry_list()])

        with open(save_path + 'object_poses.pickle', 'rb') as handle:
            object_poses= pickle.load(handle) 
        # Assign the Best Grasp Pose on Each Object
        grasp_poses, remain_gg = self.assign_grasp_pose(gg, object_poses)
        if self.debug and pose_method == 'icp':
            o3d.visualization.draw_geometries([full_pcd, *remain_gg])
        return object_poses, grasp_poses

    def get_response(self, object_list):
        '''
        Generating the defined ros perception message given the targe object list.

        Args:
            object_list(list): strings of object names.

        Returns:
            dict: both object and grasp poses which is close to the ros msg format.
        '''
        object_poses, grasp_poses = self.percept(
            object_list = object_list,
            pose_method = self.config['pose_method']
        )

        #####################################################
        # format of response_poses:
        # -------graspable
        #     |
        #     ---object_pose
        #     |  |
        #     |  |--x
        #     |  |
        #     |  |--y
        #     |  |
        #     |  ...
        #     |  |
        #     |  ---qw
        #     |
        #     ---grasp_pose (exists if graspable == True)
        #        |
        #        |--x
        #        |
        #        |--y
        #        |
        #        ...
        #        |
        #        ---qw
        #####################################################
        ### the keys of response_poses are object names.
        response_poses = dict()
        for object_name in object_poses.keys():
            response_poses[object_name] = dict()
            qw, qx, qy, qz = mat2quat(object_poses[object_name]['pose'][:3,:3])
            response_poses[object_name]['object_pose'] = {
                'x': object_poses[object_name]['pose'][0, 3],
                'y': object_poses[object_name]['pose'][1, 3],
                'z': object_poses[object_name]['pose'][2, 3],
                'qw': qw,
                'qx': qx,
                'qy': qy,
                'qz': qz
            }
            if grasp_poses[object_name] is None:
                response_poses[object_name]['graspable'] = False
            else:
                response_poses[object_name]['graspable'] = True
                response_poses[object_name]['grasp_pose'] = grasp_poses[object_name]
        print('perception finished')    
        return response_poses
