import argparse
import os
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import pyrealsense2 as rs
import rclpy


from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Header
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from models.experimental import attempt_load
from utils.general import (
    check_img_size, non_max_suppression, scale_coords,
    xyxy2xywh, plot_one_box, set_logging)
from utils.torch_utils import select_device, time_synchronized
from utils.datasets import letterbox

class D455DetectionNode(Node):
    def __init__(self):
        super().__init__('d455_detection_node')
        
        # ROS 2 参数声明
        self.declare_parameters(
            namespace='',
            parameters=[
                ('weights', 'yolov5m.pt'),
                ('img_size', 640),
                ('conf_thres', 0.25),
                ('iou_thres', 0.45),
                ('device', ''),
                ('view_img', False),
                ('save_dir', 'inference/output'),
            ]
        )
        
        # 初始化参数
        self.opt = self.parse_parameters()
        
        # 初始化ROS组件
        self.bridge = CvBridge()
        self.detection_pub = self.create_publisher(Detection2DArray, 'detections', 10)
        self.vis_pub = self.create_publisher(Image, 'detection_visualization', 10)
        
        # 初始化检测组件
        self.device = select_device(self.opt.device)
        self.half = self.device.type != 'cpu'
        self.model = attempt_load(self.opt.weights, map_location=self.device)
        self.stride = int(self.model.stride.max())
        self.imgsz = check_img_size(self.opt.img_size, s=self.stride)
        if self.half:
            self.model.half()
            
        # RealSense 初始化
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        
        # 启动检测定时器
        self.timer = self.create_timer(0.1, self.detect_callback)
        
    def parse_parameters(self):
        class Args:
            pass
        args = Args()
        args.weights = self.get_parameter('weights').value
        args.img_size = self.get_parameter('img_size').value
        args.conf_thres = self.get_parameter('conf_thres').value
        args.iou_thres = self.get_parameter('iou_thres').value
        args.device = self.get_parameter('device').value
        args.view_img = self.get_parameter('view_img').value
        args.save_dir = self.get_parameter('save_dir').value
        args.augment = False
        args.classes = None
        args.agnostic_nms = False
        return args

    def detect_callback(self):
        try:
            # 获取帧数据
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                return
                
            # 转换为 numpy 数组
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            # 预处理
            img = letterbox(color_image, new_shape=self.imgsz)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
            img = np.ascontiguousarray(img, dtype=np.float16 if self.half else np.float32)
            img /= 255.0
            
            # 转换为 Tensor
            img = torch.from_numpy(img).to(self.device)
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            
            # 推理
            pred = self.model(img, augment=self.opt.augment)[0]
            pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres)
            
            # 创建ROS消息
            detections_msg = Detection2DArray()
            header = Header(stamp=self.get_clock().now().to_msg())
            
            for det in pred[0]:  # 每帧的检测结果
                if det is not None and len(det):
                    # 转换坐标
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], color_image.shape).round()
                    
                    # 创建单个检测结果
                    detection = Detection2D()
                    bbox = xyxy2xywh(det[:, :4].view(-1, 4))
                    
                    # 边界框信息
                    detection.bbox.center.x = float(bbox[0][0])
                    detection.bbox.center.y = float(bbox[0][1])
                    detection.bbox.size_x = float(bbox[0][2])
                    detection.bbox.size_y = float(bbox[0][3])
                    
                    # 类别和置信度
                    hypothesis = ObjectHypothesisWithPose()
                    hypothesis.hypothesis.class_id = int(det[0][5])
                    hypothesis.hypothesis.score = float(det[0][4])
                    detection.results.append(hypothesis)
                    
                    # 深度计算
                    mid_pos = [(det[0][0] + det[0][2])/2, (det[0][1] + det[0][3])/2]
                    depth = depth_frame.get_distance(int(mid_pos[0]), int(mid_pos[1]))
                    detection.source_cloud.header = header
                    detection.source_cloud.point.x = mid_pos[0]
                    detection.source_cloud.point.y = mid_pos[1]
                    detection.source_cloud.point.z = depth
                    
                    detections_msg.detections.append(detection)
                    
                    # 可视化绘制
                    label = f"{self.model.names[int(det[0][5])]} {depth:.2f}m"
                    plot_one_box(det[0][:4], color_image, label=label, line_thickness=2)
            
            # 发布消息
            detections_msg.header = header
            self.detection_pub.publish(detections_msg)
            
            # 发布可视化图像
            vis_msg = self.bridge.cv2_to_imgmsg(color_image, encoding="bgr8")
            vis_msg.header = header
            self.vis_pub.publish(vis_msg)
            
            # 本地显示
            if self.opt.view_img:
                cv2.imshow("Detection", color_image)
                cv2.waitKey(1)
                
        except Exception as e:
            self.get_logger().error(f"Detection error: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = D455DetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pipeline.stop()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()