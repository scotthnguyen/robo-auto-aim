import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    pkg = get_package_share_directory('sn_auto_aim')
    detector_params = os.path.join(pkg, 'config', 'armor_detector.yaml')
    tracker_params = os.path.join(pkg, 'config', 'armor_tracker.yaml')

    use_sim_time = LaunchConfiguration('use_sim_time')

    container = ComposableNodeContainer(
        name='auto_aim_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            ComposableNode(
                package='armor_detector',
                plugin='sn_auto_aim::ArmorDetectorNode',
                name='armor_detector',
                parameters=[detector_params, {'use_sim_time': use_sim_time}],
                extra_arguments=[{'use_intra_process_comms': True}],
            ),
            ComposableNode(
                package='armor_tracker',
                plugin='sn_auto_aim::ArmorTrackerNode',
                name='armor_tracker',
                parameters=[tracker_params, {'use_sim_time': use_sim_time}],
                extra_arguments=[{'use_intra_process_comms': True}],
            ),
        ],
        output='screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation clock',
        ),
        container,
    ])
