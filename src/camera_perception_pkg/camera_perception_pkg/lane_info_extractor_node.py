import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy

from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from interfaces_pkg.msg import TargetPoint, LaneInfo, DetectionArray, BoundingBox2D, Detection
from .lib import camera_perception_func_lib as CPFL

#---------------Variable Setting---------------
# Subscribe할 토픽 이름
SUB_TOPIC_NAME = "detections"

# Publish할 토픽 이름
PUB_TOPIC_NAME = "yolov8_lane_info"
ROI_IMAGE_TOPIC_NAME = "roi_image"  # 추가: ROI 이미지 퍼블리시 토픽

# 화면에 이미지를 처리하는 과정을 띄울것인지 여부: True, 또는 False 중 택1하여 입력
SHOW_IMAGE = True
#----------------------------------------------


class Yolov8InfoExtractor(Node):
    def __init__(self):
        super().__init__('lane_info_extractor_node')

        self.sub_topic = self.declare_parameter('sub_detection_topic', SUB_TOPIC_NAME).value
        self.pub_topic = self.declare_parameter('pub_topic', PUB_TOPIC_NAME).value
        self.show_image = self.declare_parameter('show_image', SHOW_IMAGE).value
        # 버드아이뷰 변환을 위한 원본 및 목적 좌표 (launch 파일에서 튜닝)
        # 튜닝 팁: src_mat은 원본 이미지의 차선 소실점 근처 사다리꼴, dst_mat은 변환 후 직사각형
        self.declare_parameter('perspective.src_points', [173, 132, 422, 134, 632, 241, 5, 225])
        self.declare_parameter('perspective.dst_points_ratio', [0.3, 0.0, 0.7, 0.0, 0.7, 1.0, 0.3, 1.0])
        
        # ROI 및 차선 관련 파라미터
        self.declare_parameter('roi.cutting_idx', 300)
        self.declare_parameter('lane.width', 300)
        self.declare_parameter('lane.detection_thickness', 10)
        self.declare_parameter('lane.theta_limit', 70)

        # --- 파라미터 값 가져오기 ---
        self.sub_topic = self.get_parameter('sub_detection_topic').value
        self.pub_topic = self.get_parameter('pub_topic').value
        self.show_image = self.get_parameter('show_image').value
        
        src_points_list = self.get_parameter('perspective.src_points').value
        self.src_mat = np.array(src_points_list).reshape(4, 2).tolist()

        self.dst_points_ratio = self.get_parameter('perspective.dst_points_ratio').value
        self.cutting_idx = self.get_parameter('roi.cutting_idx').value
        self.lane_width = self.get_parameter('lane.width').value
        self.detection_thickness = self.get_parameter('lane.detection_thickness').value
        self.theta_limit = self.get_parameter('lane.theta_limit').value

        self.cv_bridge = CvBridge()

        # QoS settings
        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )
        
        self.subscriber = self.create_subscription(DetectionArray, self.sub_topic, self.yolov8_detections_callback, self.qos_profile)
        self.publisher = self.create_publisher(LaneInfo, self.pub_topic, self.qos_profile)

        # ROI 이미지 퍼블리셔 추가
        self.roi_image_publisher = self.create_publisher(Image, ROI_IMAGE_TOPIC_NAME, self.qos_profile)


    def fit_polynomial_and_get_waypoints(self, edge_image):
        """
        엣지 이미지에서 차선 픽셀을 찾아 2차 다항식을 피팅하고,
        중앙 경로(waypoints)를 계산합니다.
        """
        # 이미지의 y, x 좌표 인덱스를 가져옴 (차선 픽셀 위치 찾기)
        nonzero = edge_image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # 최소 픽셀 수 미만이면 계산하지 않음
        if len(nonzerox) < 50:
            return None, None

        # 2차 다항식 피팅: x = ay^2 + by + c
        # y를 기준으로 x를 예측하는 모델을 만듭니다.
        fit_params = np.polyfit(nonzeroy, nonzerox, 2)

        # y값(이미지 높이) 범위를 생성
        ploty = np.linspace(0, edge_image.shape[0]-1, edge_image.shape[0])

        # 생성된 y값에 대한 x값 예측 (피팅된 곡선)
        try:
            fit_x = fit_params[0]*ploty**2 + fit_params[1]*ploty + fit_params[2]
            return fit_x.astype(int), ploty.astype(int)
        except TypeError:
            return None, None

    def yolov8_detections_callback(self, detection_msg: DetectionArray):
        if len(detection_msg.detections) == 0:
            return
        
        # --- 1. 좌/우 차선 분리 ---
        # 이미지 중앙 x 좌표
        img_center_x = detection_msg.detections[0].mask.width // 2
        
        left_lane_detections = [d for d in detection_msg.detections if d.class_name == 'lane' and d.bbox.center.position.x < img_center_x]
        right_lane_detections = [d for d in detection_msg.detections if d.class_name == 'lane' and d.bbox.center.position.x >= img_center_x]
        
        # 좌/우 차선에 대한 DetectionArray 메시지 생성
        left_detection_array = DetectionArray(header=detection_msg.header, detections=left_lane_detections)
        right_detection_array = DetectionArray(header=detection_msg.header, detections=right_lane_detections)

        # --- 2. 엣지 이미지 생성 ---
        left_lane_edge = CPFL.draw_edges_from_detections(left_detection_array, color=255)
        right_lane_edge = CPFL.draw_edges_from_detections(right_detection_array, color=255)
        
        # 두 엣지 이미지를 합쳐서 전체 차선 이미지 생성
        combined_lane_edge = cv2.bitwise_or(left_lane_edge, right_lane_edge)

        if combined_lane_edge is None:
            self.get_logger().warn("Failed to create lane edge image.")
            return

        # --- 3. 버드아이뷰 변환 ---
        (h, w) = (combined_lane_edge.shape[0], combined_lane_edge.shape[1])
        
        # dst_mat을 이미지 크기와 비율에 따라 동적으로 계산
        r = self.dst_points_ratio
        dst_mat = [[round(w * r[0]), round(h * r[1])], [round(w * r[2]), round(h * r[3])], 
                   [round(w * r[4]), round(h * r[5])], [round(w * r[6]), round(h * r[7])]]

        lane_bird_image = CPFL.bird_convert(combined_lane_edge, srcmat=self.src_mat, dstmat=dst_mat)
        
        # --- 4. ROI 추출 ---
        roi_image = CPFL.roi_rectangle_below(lane_bird_image, cutting_idx=self.cutting_idx)

        if self.show_image:
            cv2.imshow('Left Lane Edge', left_lane_edge)
            cv2.imshow('Right Lane Edge', right_lane_edge)
            cv2.imshow('Combined Bird-eye View', lane_bird_image)
            cv2.imshow('ROI Image', roi_image)
            cv2.waitKey(1)
        
        # ROI 이미지를 ROS 메시지로 변환하여 퍼블리시
        try:
            roi_image_msg = self.cv_bridge.cv2_to_imgmsg(roi_image.astype(np.uint8), encoding="mono8")
            self.roi_image_publisher.publish(roi_image_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to convert and publish ROI image: {e}")
        
        left_fitx, ploty = self.fit_polynomial_and_get_waypoints(left_lane_edge_bird_roi) # 버드아이뷰 변환 및 ROI가 적용된 좌측 차선 이미지
        right_fitx, _ = self.fit_polynomial_and_get_waypoints(right_lane_edge_bird_roi) # 버드아이뷰 변환 및 ROI가 적용된 우측 차선 이미지

        target_points = []
        center_fitx = None

        if left_fitx is not None and right_fitx is not None:
            # 양쪽 차선이 모두 인식되면, 중앙 경로 계산
            center_fitx = (left_fitx + right_fitx) // 2
        elif left_fitx is not None:
            # 왼쪽 차선만 인식되면, 차선 폭 만큼 오른쪽으로 오프셋
            center_fitx = left_fitx + (self.lane_width // 2)
        elif right_fitx is not None:
            # 오른쪽 차선만 인식되면, 차선 폭 만큼 왼쪽으로 오프셋
            center_fitx = right_fitx - (self.lane_width // 2)

        if center_fitx is not None:
            # Waypoints 생성 (예: 10개)
            # ploty 배열에서 일정한 간격으로 인덱스를 샘플링
            for i in range(0, len(ploty), len(ploty) // 10):
                target_point = TargetPoint()
                target_point.target_x = int(center_fitx[i])
                target_point.target_y = int(ploty[i])
                target_points.append(target_point)

        lane = LaneInfo()
        lane.slope = grad
        lane.target_points = target_points

        self.publisher.publish(lane)


def main(args=None):
    rclpy.init(args=args)
    node = Yolov8InfoExtractor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\nshutdown\n\n")
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()
  
if __name__ == '__main__':
    main()
