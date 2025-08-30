import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy

from std_msgs.msg import String, Bool
from interfaces_pkg.msg import PathPlanningResult, DetectionArray, MotionCommand
from .lib import decision_making_func_lib as DMFL

#---------------Variable Setting---------------
SUB_DETECTION_TOPIC_NAME = "detections"
SUB_PATH_TOPIC_NAME = "path_planning_result"
SUB_TRAFFIC_LIGHT_TOPIC_NAME = "yolov8_traffic_light_info"
SUB_LIDAR_OBSTACLE_TOPIC_NAME = "lidar_obstacle_info"
PUB_TOPIC_NAME = "topic_control_signal"

#----------------------------------------------

# 모션 플랜 발행 주기 (초) - 소수점 필요 (int형은 반영되지 않음)
TIMER = 0.1

class MotionPlanningNode(Node):
    def __init__(self):
        super().__init__('motion_planner_node')

        # 토픽 이름 설정
        self.sub_detection_topic = self.declare_parameter('sub_detection_topic', SUB_DETECTION_TOPIC_NAME).value
        self.sub_path_topic = self.declare_parameter('sub_lane_topic', SUB_PATH_TOPIC_NAME).value
        self.sub_traffic_light_topic = self.declare_parameter('sub_traffic_light_topic', SUB_TRAFFIC_LIGHT_TOPIC_NAME).value
        self.sub_lidar_obstacle_topic = self.declare_parameter('sub_lidar_obstacle_topic', SUB_LIDAR_OBSTACLE_TOPIC_NAME).value
        self.pub_topic = self.declare_parameter('pub_topic', PUB_TOPIC_NAME).value
        
        self.timer_period = self.declare_parameter('timer', TIMER).value

        # QoS 설정
        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        # 변수 초기화
        self.detection_data = None
        self.path_data = None
        self.traffic_light_data = None
        self.lidar_data = None
        '''
        # --- PID 제어 변수 추가 ---
        # 튜닝이 필요한 PID 게인 값 (초기값이며, 주행 테스트를 통해 조절)
        self.kp = self.declare_parameter('pid.kp', 0.03).value
        self.ki = self.declare_parameter('pid.ki', 0.001).value
        self.kd = self.declare_parameter('pid.kd', 0.01).value
        
        self.prev_error = 0.0
        self.integral = 0.0
        # ---------------------------
        '''

        self.steering_command = 0
        self.left_speed_command = 0
        self.right_speed_command = 0
        


        # 서브스크라이버 설정
        self.detection_sub = self.create_subscription(DetectionArray, self.sub_detection_topic, self.detection_callback, self.qos_profile)
        self.path_sub = self.create_subscription(PathPlanningResult, self.sub_path_topic, self.path_callback, self.qos_profile)
        self.traffic_light_sub = self.create_subscription(String, self.sub_traffic_light_topic, self.traffic_light_callback, self.qos_profile)
        self.lidar_sub = self.create_subscription(Bool, self.sub_lidar_obstacle_topic, self.lidar_callback, self.qos_profile)

        # 퍼블리셔 설정
        self.publisher = self.create_publisher(MotionCommand, self.pub_topic, self.qos_profile)

        # 타이머 설정
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    def detection_callback(self, msg: DetectionArray):
        self.detection_data = msg

    def path_callback(self, msg: PathPlanningResult):
        self.path_data = list(zip(msg.x_points, msg.y_points))
                
    def traffic_light_callback(self, msg: String):
        self.traffic_light_data = msg

    def lidar_callback(self, msg: Bool):
        self.lidar_data = msg
        
    def timer_callback(self):

        if self.lidar_data is not None and self.lidar_data.data is True:
            # 라이다가 장애물을 감지한 경우
            self.steering_command = 0 
            self.left_speed_command = 0 
            self.right_speed_command = 0 

        elif self.traffic_light_data is not None and self.traffic_light_data.data == 'Red':
            # 빨간색 신호등을 감지한 경우
            for detection in self.detection_data.detections:
                if detection.class_name=='traffic_light':
                    x_min = int(detection.bbox.center.position.x - detection.bbox.size.x / 2) # bbox의 좌측상단 꼭짓점 x좌표
                    x_max = int(detection.bbox.center.position.x + detection.bbox.size.x / 2) # bbox의 우측하단 꼭짓점 x좌표
                    y_min = int(detection.bbox.center.position.y - detection.bbox.size.y / 2) # bbox의 좌측상단 꼭짓점 y좌표
                    y_max = int(detection.bbox.center.position.y + detection.bbox.size.y / 2) # bbox의 우측하단 꼭짓점 y좌표

                    if y_max < 150:
                        # 신호등 위치에 따른 정지명령 결정
                        self.steering_command = 0 
                        self.left_speed_command = 0 
                        self.right_speed_command = 0
        else:
            if self.path_data is None:
                self.steering_command = 0
            else:
                '''
                # 1. 목표 지점(Look-ahead point) 설정
                # path_data의 끝부분이 차량과 가장 가까운 경로임
                # 너무 가까운 점은 불안정하므로, 약간 앞쪽의 점을 목표로 삼음 (예: 뒤에서 20번째 점)
                # 이 값은 차량 속도나 반응성에 따라 튜닝이 필요
                look_ahead_idx = max(0, len(self.path_data) - 20) 
                target_x = self.path_data[look_ahead_idx][0]

                # 2. 횡방향 오차(CTE) 계산
                # 차량의 현재 x 위치는 이미지의 중앙인 320으로 가정
                current_x = 320
                error = target_x - current_x

                # 3. PID 제어 계산
                # 비례항 (Proportional)
                p_term = self.kp * error

                # 적분항 (Integral) - 오차 누적
                self.integral += error * self.timer_period
                i_term = self.ki * self.integral

                # 미분항 (Derivative) - 오차의 변화율
                derivative = (error - self.prev_error) / self.timer_period
                d_term = self.kd * derivative
                self.prev_error = error

                # 4. 최종 조향값 계산
                # PID 제어값을 합산하여 조향 제어량 산출
                pid_output = p_term + i_term + d_term

                # 5. 조향 명령으로 변환
                # pid_output 값을 차량이 사용하는 조향각 범위(-7 ~ 7)로 매핑
                # 이 범위는 실험을 통해 조절해야 함
                self.steering_command = int(max(-7, min(7, pid_output)))
                '''

                target_slope = DMFL.calculate_slope_between_points(self.path_data[-10], self.path_data[-1])
                
                # 조향 각도 범위 세분화
                if target_slope > 0: # right
                    if 0 < target_slope <= 5:
                        self.steering_command = 0
                    elif 5 < target_slope <= 10:
                        self.steering_command = 1
                    elif 10 < target_slope <= 15:
                        self.steering_command = 2
                    elif 15 < target_slope <= 20:
                        self.steering_command = 3
                    elif 20 < target_slope <= 30:
                        self.steering_command = 4
                    elif 30 < target_slope <= 50:
                        self.steering_command = 5
                    elif 50 < target_slope <= 70:
                        self.steering_command = 6
                    elif target_slope > 70:
                        self.steering_command = 7
                    else:
                        self.steering_command = 0
                elif target_slope < 0: # left
                    if -5 <= target_slope < 0:
                        self.steering_command = 0
                    elif -10 <= target_slope < -5:
                        self.steering_command = -1
                    elif -15 <= target_slope < -10:
                        self.steering_command = -2
                    elif -20 <= target_slope < -15:
                        self.steering_command = -3
                    elif -30 <= target_slope < -20:
                        self.steering_command = -4
                    elif -50 <= target_slope < -30:
                        self.steering_command = -5
                    elif -70 <= target_slope < -50:
                        self.steering_command = -6
                    elif target_slope < -70:
                        self.steering_command = -7
                    else:
                        self.steering_command = 0
                else: # target_slope == 0
                    self.steering_command = 0


            self.left_speed_command = 100  # 예시 속도 값 (255가 최대 속도)
            self.right_speed_command = 100  # 예시 속도 값 (255가 최대 속도)



        self.get_logger().info(f"steering: {self.steering_command}, " 
                               f"left_speed: {self.left_speed_command}, " 
                               f"right_speed: {self.right_speed_command}")

        # 모션 명령 메시지 생성 및 퍼블리시
        motion_command_msg = MotionCommand()
        motion_command_msg.steering = self.steering_command
        motion_command_msg.left_speed = self.left_speed_command
        motion_command_msg.right_speed = self.right_speed_command
        self.publisher.publish(motion_command_msg)

def main(args=None):
    rclpy.init(args=args)
    node = MotionPlanningNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\nshutdown\n\n")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
