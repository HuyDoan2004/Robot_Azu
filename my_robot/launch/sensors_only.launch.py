from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.parameter_descriptions import ParameterFile
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_share = get_package_share_directory('my_robot')

    cam_cfg   = PathJoinSubstitution([pkg_share, 'configs', 'realsense.yaml'])
    lidar_cfg = PathJoinSubstitution([pkg_share, 'configs', 'rplidar.yaml'])

    # Lấy offset gắn camera từ YAML (cam_tx/ty/tz)
    cam_tx = LaunchConfiguration('cam_tx', default='0.0')
    cam_ty = LaunchConfiguration('cam_ty', default='0.0')
    cam_tz = LaunchConfiguration('cam_tz', default='0.30')

    cam = Node(
        package='my_robot', executable='realsense_yolo_node',
        name='realsense_yolo_node', output='screen', emulate_tty=True, respawn=True,
        parameters=[ParameterFile(cam_cfg, allow_substs=True)]
    )

    lidar = Node(
        package='my_robot', executable='realsense_lidar_node',
        name='realsense_lidar_node', output='screen', emulate_tty=True, respawn=True,
        parameters=[ParameterFile(lidar_cfg, allow_substs=True)]
    )

    # TF: base_link -> camera_link (vị trí gắn cam trên robot)
    tf_base2cam = Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='tf_base_to_camera',
        arguments=[cam_tx, cam_ty, cam_tz, '0', '0', '0', 'base_link', 'camera_link'],
        output='screen'
    )

    # TF: camera_link -> camera_color_optical_frame (chuẩn optical: x=right,y=down,z=forward)
    yaw  = str(-3.1415926535/2)   # -pi/2
    roll = str(-3.1415926535/2)   # -pi/2
    tf_cam2color = Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='tf_camera_to_color_optical',
        arguments=['0','0','0', yaw, '0', roll, 'camera_link', 'camera_color_optical_frame'],
        output='screen'
    )

    # TF: camera_link -> camera_depth_optical_frame
    tf_cam2depth = Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='tf_camera_to_depth_optical',
        arguments=['0','0','0', yaw, '0', roll, 'camera_link', 'camera_depth_optical_frame'],
        output='screen'
    )

    # Static TF cho LiDAR như cũ
    static_laser = Node(
        package='tf2_ros', executable='static_transform_publisher',
        arguments=['--x','0','--y','0','--z','0.05','--roll','0','--pitch','0','--yaw','0',
                   '--frame-id','base_link','--child-frame-id','laser']
    )

    return LaunchDescription([
        cam, lidar,
        tf_base2cam, tf_cam2color, tf_cam2depth,
        static_laser
    ])
