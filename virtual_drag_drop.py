import cv2
import mediapipe as mp
import time
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from enum import Enum
import random

@dataclass
class Config:
    """虚拟拖拽系统配置参数"""
    MIN_DETECTION_CONFIDENCE: float = 0.7  # 手势检测置信度阈值
    MIN_TRACKING_CONFIDENCE: float = 0.5   # 手势跟踪置信度阈值
    MAX_NUM_HANDS: int = 2                 # 最大检测手数
    DRAG_DISTANCE_THRESHOLD: int = 100     # 拖拽距离阈值
    DEFAULT_SHAPE_SIZE: int = 150          # 默认形状大小
    DEFAULT_ALPHA: float = 0.6             # 默认透明度
    ROTATION_SCALE: float = 0.5            # 旋转灵敏度
    SCALE_FACTOR: float = 1.0              # 缩放灵敏度
    TRAIL_LENGTH: int = 10                 # 轨迹长度
    MIN_SCALE: float = 0.2                 # 最小缩放比例
    MAX_SCALE: float = 5.0                 # 最大缩放比例
    SCALE_THRESHOLD: int = 50              # 缩放激活阈值
    FIST_THRESHOLD: int = 100              # 握拳检测阈值
    INITIAL_POSITIONS = [                  # 形状初始位置
        (200, 300),
        (400, 300),
        (600, 300)
    ]

class ShapeType(Enum):
    """形状类型枚举"""
    RECTANGLE = 0
    CIRCLE = 1
    TRIANGLE = 2
    STAR = 3

class Shape:
    """形状基类"""
    def __init__(self, x: int, y: int, size: int, shape_type: ShapeType):
        self.x = x                      # 中心x坐标
        self.y = y                      # 中心y坐标
        self.initial_x = x              # 初始x坐标
        self.initial_y = y              # 初始y坐标
        self.size = size                # 大小
        self.shape_type = shape_type    # 形状类型
        self.rotation = 0               # 旋转角度
        self.scale = 1.0               # 缩放比例
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))  # 随机颜色
        self.alpha = 0.6               # 透明度
        self.trail = []                # 运动轨迹
        self.initial_size = size       # 初始大小

    def reset(self):
        """重置形状到初始状态"""
        self.x = self.initial_x
        self.y = self.initial_y
        self.rotation = 0
        self.scale = 1.0
        self.trail = []

    def contains_point(self, point_x: int, point_y: int) -> bool:
        """检查点是否在形状内"""
        if self.shape_type == ShapeType.CIRCLE:
            return math.hypot(point_x - self.x, point_y - self.y) <= self.size * self.scale / 2
        elif self.shape_type == ShapeType.RECTANGLE:
            half_size = self.size * self.scale / 2
            return (abs(point_x - self.x) <= half_size and 
                   abs(point_y - self.y) <= half_size)
        elif self.shape_type == ShapeType.TRIANGLE:
            # 计算三角形的三个顶点
            half_size = int(self.size * self.scale / 2)
            # 三角形的三个顶点
            top = (self.x, self.y - half_size)
            bottom_left = (self.x - half_size, self.y + half_size)
            bottom_right = (self.x + half_size, self.y + half_size)
            
            # 使用面积法判断点是否在三角形内
            def area(x1, y1, x2, y2, x3, y3):
                return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)
            
            # 计算三角形总面积
            A = area(top[0], top[1], bottom_left[0], bottom_left[1], bottom_right[0], bottom_right[1])
            
            # 计算点与三角形顶点构成的三个小三角形的面积
            A1 = area(point_x, point_y, bottom_left[0], bottom_left[1], bottom_right[0], bottom_right[1])
            A2 = area(top[0], top[1], point_x, point_y, bottom_right[0], bottom_right[1])
            A3 = area(top[0], top[1], bottom_left[0], bottom_left[1], point_x, point_y)
            
            # 如果三个小三角形的面积之和等于大三角形的面积，则点在三角形内
            return abs(A - (A1 + A2 + A3)) < 1  # 添加一个小的误差范围

        return False

    def draw(self, image: np.ndarray, is_active: bool = False) -> np.ndarray:
        """绘制形状"""
        overlay = image.copy()
        color = (255, 0, 255) if is_active else self.color
        
        # 绘制轨迹
        if len(self.trail) > 1:
            for i in range(len(self.trail) - 1):
                alpha = (i + 1) / len(self.trail) * 0.5
                cv2.line(overlay, self.trail[i], self.trail[i + 1], color, 2)
                image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

        # 根据形状类型绘制不同的形状
        if self.shape_type == ShapeType.CIRCLE:
            cv2.circle(overlay, (self.x, self.y), int(self.size * self.scale / 2), color, -1)
        elif self.shape_type == ShapeType.RECTANGLE:
            # 计算旋转后的矩形顶点
            half_size = int(self.size * self.scale / 2)
            points = np.array([
                [-half_size, -half_size],
                [half_size, -half_size],
                [half_size, half_size],
                [-half_size, half_size]
            ], dtype=np.float32)
            
            # 应用旋转
            angle = math.radians(self.rotation)
            rotation_matrix = np.array([
                [math.cos(angle), -math.sin(angle)],
                [math.sin(angle), math.cos(angle)]
            ])
            points = np.dot(points, rotation_matrix)
            
            # 移动到中心位置
            points = points + np.array([self.x, self.y])
            points = points.astype(np.int32)
            
            cv2.fillPoly(overlay, [points], color)
        elif self.shape_type == ShapeType.TRIANGLE:
            half_size = int(self.size * self.scale / 2)
            points = np.array([
                [self.x, self.y - half_size],
                [self.x - half_size, self.y + half_size],
                [self.x + half_size, self.y + half_size]
            ], dtype=np.int32)
            cv2.fillPoly(overlay, [points], color)
        
        image = cv2.addWeighted(overlay, self.alpha, image, 1 - self.alpha, 0)
        return image

    def update_position(self, x: int, y: int):
        """更新位置并记录轨迹"""
        self.trail.append((self.x, self.y))
        if len(self.trail) > Config.TRAIL_LENGTH:
            self.trail.pop(0)
        self.x = x
        self.y = y

    def update_rotation(self, angle: float):
        """更新旋转角度"""
        self.rotation += angle * Config.ROTATION_SCALE
        self.rotation = self.rotation % 360

    def update_scale(self, scale_factor: float):
        """更新缩放比例"""
        new_scale = self.scale + scale_factor * Config.SCALE_FACTOR
        self.scale = max(Config.MIN_SCALE, min(Config.MAX_SCALE, new_scale))  # 限制缩放范围

class ShapeManager:
    """形状管理器：负责创建、显示和更新可拖拽的形状"""
    
    def __init__(self):
        """初始化形状管理器"""
        self.shapes: List[Shape] = []
        self.active_index: int = -1
        self.drag_active: bool = False
        self.last_distance: float = 0
        self.last_angle: float = 0
        self.offset_x: float = 0
        self.offset_y: float = 0
        self.initial_scale: float = 1.0  # 记录开始缩放时的初始比例

    def create(self, x: int, y: int, size: int, shape_type: ShapeType) -> None:
        """创建一个新的形状"""
        self.shapes.append(Shape(x, y, size, shape_type))

    def display(self, image: np.ndarray) -> np.ndarray:
        """显示所有形状"""
        for i, shape in enumerate(self.shapes):
            image = shape.draw(image, i == self.active_index)
        return image

    def check_collision(self, x: int, y: int) -> int:
        """检查点是否在某个形状内"""
        for i, shape in enumerate(self.shapes):
            if shape.contains_point(x, y):
                self.active_index = i
                return i
        return -1

    def set_offset(self, x: int, y: int) -> None:
        """设置拖拽偏移量"""
        if self.active_index != -1:
            shape = self.shapes[self.active_index]
            self.offset_x = x - shape.x
            self.offset_y = y - shape.y

    def update_shape(self, x: int, y: int, distance: float = None, angle: float = None) -> None:
        """更新形状状态"""
        if self.active_index != -1:
            shape = self.shapes[self.active_index]
            
            # 更新位置（如果提供了新位置）
            if x is not None and y is not None:
                new_x = int(x - self.offset_x)
                new_y = int(y - self.offset_y)
                shape.update_position(new_x, new_y)
            
            # 更新旋转（如果提供了新角度）
            if angle is not None and self.last_angle is not None:
                angle_diff = angle - self.last_angle
                shape.update_rotation(angle_diff)
            
            # 更新缩放（如果提供了新距离）
            if distance is not None and self.last_distance > 0:
                scale_ratio = distance / self.last_distance
                new_scale = shape.scale * scale_ratio
                shape.scale = max(Config.MIN_SCALE, min(Config.MAX_SCALE, new_scale))
            
            # 更新状态
            if distance is not None:
                self.last_distance = distance
            if angle is not None:
                self.last_angle = angle

    def reset_all_shapes(self):
        """重置所有形状到初始状态"""
        for shape in self.shapes:
            shape.reset()
        self.active_index = -1
        self.drag_active = False
        self.last_distance = 0
        self.last_angle = 0

class VirtualDragController:
    """虚拟拖拽控制器：负责手势识别和拖拽控制"""

    def __init__(self):
        """初始化虚拟拖拽控制器"""
        self.config = Config()
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        self.image: Optional[np.ndarray] = None

    def _process_landmarks(self, hand_landmarks, resize_w: int, resize_h: int) -> Tuple[List[Tuple[int, int]], float, float, float, bool]:
        """
        处理手部关键点
        Args:
            hand_landmarks: MediaPipe手部关键点数据
            resize_w: 图像宽度
            resize_h: 图像高度
        Returns:
            手指坐标列表[大拇指，食指，中指]，食指中指距离，大拇指食指距离，手指角度，是否握拳
        """
        landmark_list = []
        
        for landmark_id, finger_axis in enumerate(hand_landmarks.landmark):
            landmark_list.append([
                landmark_id,
                math.ceil(finger_axis.x * resize_w),
                math.ceil(finger_axis.y * resize_h)
            ])

        # 获取手掌基部和手指尖点
        wrist = (landmark_list[0][1], landmark_list[0][2])          # 手腕
        thumb_tip = (landmark_list[4][1], landmark_list[4][2])      # 大拇指尖
        index_tip = (landmark_list[8][1], landmark_list[8][2])      # 食指尖
        middle_tip = (landmark_list[12][1], landmark_list[12][2])   # 中指尖
        ring_tip = (landmark_list[16][1], landmark_list[16][2])     # 无名指尖
        pinky_tip = (landmark_list[20][1], landmark_list[20][2])    # 小指尖
        palm_center = (landmark_list[9][1], landmark_list[9][2])    # 手掌中心

        # 检测握拳手势
        finger_tips = [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]
        distances_to_palm = [
            math.hypot(tip[0] - palm_center[0], tip[1] - palm_center[1])
            for tip in finger_tips
        ]
        is_fist = all(d < Config.FIST_THRESHOLD for d in distances_to_palm)
        
        # 计算食指和中指之间的距离（用于拖拽）
        drag_distance = math.hypot(middle_tip[0] - index_tip[0], middle_tip[1] - index_tip[1])
        
        # 计算大拇指和食指之间的距离（用于缩放）
        scale_distance = math.hypot(index_tip[0] - thumb_tip[0], index_tip[1] - thumb_tip[1])
        
        # 计算食指和中指的角度（用于旋转）
        angle = math.degrees(math.atan2(middle_tip[1] - index_tip[1], middle_tip[0] - index_tip[0]))
        
        return [thumb_tip, index_tip, middle_tip], drag_distance, scale_distance, angle, is_fist

    def recognize(self):
        """主要识别和控制循环"""
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise Exception("Cannot open camera")

            resize_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) * 2
            resize_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) * 2
            
            # 初始化形状管理器并创建测试形状
            shapeManager = ShapeManager()
            shape_types = [ShapeType.RECTANGLE, ShapeType.CIRCLE, ShapeType.TRIANGLE]
            for i, shape_type in enumerate(shape_types):
                pos = Config.INITIAL_POSITIONS[i]
                shapeManager.create(pos[0], pos[1], self.config.DEFAULT_SHAPE_SIZE, shape_type)

            with self.mp_hands.Hands(
                min_detection_confidence=self.config.MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=self.config.MIN_TRACKING_CONFIDENCE,
                max_num_hands=self.config.MAX_NUM_HANDS
            ) as hands:
                fpsTime = time.time()
                scaling_mode = False  # 是否处于缩放模式
                
                while cap.isOpened():
                    success, self.image = cap.read()
                    if not success:
                        print("Cannot get camera frame")
                        continue

                    # 图像预处理
                    self.image = cv2.resize(self.image, (resize_w, resize_h))
                    self.image.flags.writeable = False
                    self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
                    self.image = cv2.flip(self.image, 1)
                    
                    # 手势识别
                    results = hands.process(self.image)
                    self.image.flags.writeable = True
                    self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)

                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            # 绘制手部关键点
                            self.mp_drawing.draw_landmarks(
                                self.image,
                                hand_landmarks,
                                self.mp_hands.HAND_CONNECTIONS,
                                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                                self.mp_drawing_styles.get_default_hand_connections_style()
                            )

                            # 处理手部关键点
                            finger_points, drag_distance, scale_distance, angle, is_fist = self._process_landmarks(
                                hand_landmarks, resize_w, resize_h
                            )
                            
                            # 检测握拳手势
                            if is_fist:
                                shapeManager.reset_all_shapes()
                                scaling_mode = False
                                continue

                            thumb_point, index_point, middle_point = finger_points
                            
                            # 计算食指和中指的中点（用于拖拽）
                            drag_center = (
                                (index_point[0] + middle_point[0]) // 2,
                                (index_point[1] + middle_point[1]) // 2
                            )

                            # 绘制手指点和连线
                            for point in finger_points:
                                cv2.circle(self.image, point, 10, (255, 0, 255), -1)
                            # 绘制拖拽连线
                            cv2.line(self.image, index_point, middle_point, (255, 0, 255), 5)
                            # 绘制缩放连线
                            cv2.line(self.image, thumb_point, index_point, (0, 255, 0), 5)

                            # 处理缩放逻辑（使用大拇指和食指）
                            thumb_index_center = (
                                (thumb_point[0] + index_point[0]) // 2,
                                (thumb_point[1] + index_point[1]) // 2
                            )

                            if not scaling_mode:
                                # 检查是否可以开始缩放
                                if (scale_distance < Config.SCALE_THRESHOLD and 
                                    shapeManager.check_collision(thumb_index_center[0], thumb_index_center[1]) != -1):
                                    scaling_mode = True
                                    shapeManager.last_distance = scale_distance
                                # 如果没有在缩放，检查是否可以拖拽
                                elif (drag_distance < self.config.DRAG_DISTANCE_THRESHOLD and 
                                      shapeManager.check_collision(drag_center[0], drag_center[1]) != -1):
                                    shapeManager.drag_active = True
                                    shapeManager.set_offset(drag_center[0], drag_center[1])
                                    shapeManager.last_angle = angle
                            else:
                                # 正在缩放
                                if scale_distance > Config.SCALE_THRESHOLD * 2:
                                    # 如果手指张开太大，退出缩放模式
                                    scaling_mode = False
                                else:
                                    # 更新缩放
                                    shapeManager.update_shape(None, None, scale_distance, None)

                            # 处理拖拽逻辑
                            if shapeManager.drag_active and not scaling_mode:
                                shapeManager.update_shape(drag_center[0], drag_center[1], None, angle)
                                if drag_distance > self.config.DRAG_DISTANCE_THRESHOLD:
                                    shapeManager.drag_active = False
                                    shapeManager.active_index = -1

                    # 更新显示
                    self.image = shapeManager.display(self.image)

                    # 显示状态
                    mode_text = "Scaling" if scaling_mode else "Dragging" if shapeManager.drag_active else "None"
                    cv2.putText(self.image, f"Mode: {mode_text}", (10, 30),
                              cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

                    # 显示FPS
                    cTime = time.time()
                    fps = 1 / (cTime - fpsTime)
                    fpsTime = cTime
                    cv2.putText(self.image, f"FPS: {int(fps)}", (10, 70),
                              cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

                    # 显示操作说明
                    cv2.putText(self.image, "Controls:", (10, 120),
                              cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                    cv2.putText(self.image, "Index+Middle fingers: Drag & Drop", (30, 160),
                              cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                    cv2.putText(self.image, "Thumb+Index fingers: Scale", (30, 200),
                              cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                    cv2.putText(self.image, "Rotate fingers while dragging", (30, 240),
                              cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                    cv2.putText(self.image, "Make a fist to reset", (30, 280),
                              cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

                    # 显示画面
                    cv2.imshow('Virtual Drag and Drop', self.image)

                    if cv2.waitKey(5) & 0xFF == 27:
                        break

        except Exception as e:
            print(f"Error occurred: {str(e)}")
        finally:
            if 'cap' in locals():
                cap.release()
            cv2.destroyAllWindows()

# 程序入口
if __name__ == "__main__":
    controller = VirtualDragController()
    controller.recognize()
