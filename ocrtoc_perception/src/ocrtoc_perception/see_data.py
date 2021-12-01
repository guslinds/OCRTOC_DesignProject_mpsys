#!/usr/bin/env python

import numpy as np
import cv2
from numpy.core.numeric import full
import rospy
import rospkg
import open3d as o3d
from transforms3d.quaternions import mat2quat
import time
import copy
import os
from copy import deepcopy

import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from ocrtoc_common.camera_interface import CameraInterface
from ocrtoc_common.transform_interface import TransformInterface
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
import pickle


if __name__ == "__main__" :
    from pose.pose_6d import get_6d_pose_by_geometry, load_model_pcd
    from pose.pose_correspondence import get_pose_superglue
    rospack = rospkg.RosPack()
    save_path = os.path.join(rospack.get_path('ocrtoc_perception'),'data','')
    full_pcd = o3d.io.read_point_cloud(save_path + 'full_pcd.pcd') 
    with open(save_path +  'color_images.pickle', 'rb') as handle:
        color_images = pickle.load(handle)  
    with open(save_path + 'camera_poses.pickle', 'rb') as handle:
        camera_poses = pickle.load(handle) 
    # Compute Grasping Poses (Many Poses in a Scene)
    with open(save_path + 'gg.pickle', 'rb') as handle:
        gg = pickle.load(handle)      

    with open(save_path + 'object_poses.pickle', 'rb') as handle:
        object_poses= pickle.load(handle) 
        
        
        
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
    o3d.visualization.draw_geometries([frame, full_pcd, *gg.to_open3d_geometry_list()])
        



