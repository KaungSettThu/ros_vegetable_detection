import os
from launch import LaunchDescription
from launch.actions import TimerAction, DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription
from ament_index_python.packages import get_package_share_directory
import xacro

def generate_launch_description():

    # Launch argument for RViz
    rviz_arg = DeclareLaunchArgument('rviz', default_value='true', description='Launch RViz?')

    # Path to your RealSense XACRO
    xacro_file = os.path.join(
        get_package_share_directory('realsense2_description'),
        'urdf',
        'test_d435i_camera.urdf.xacro'
    )

    # Process XACRO to URDF
    robot_description_config = xacro.process_file(xacro_file).toxml()

    # Include Gazebo launch (with gazebo_ros_factory)
    gazebo_launch_file = os.path.join(
        get_package_share_directory('gazebo_ros'),
        'launch',
        'gazebo.launch.py'
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([gazebo_launch_file]),
        launch_arguments={'verbose': 'true'}.items()
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description_config}]
    )

    # Spawn entity in Gazebo after 3 seconds delay
    spawn_entity = TimerAction(
        period=3.0,
        actions=[
            Node(
                package='gazebo_ros',
                executable='spawn_entity.py',
                arguments=['-topic', 'robot_description', '-entity', 'realsense_camera'],
                output='screen'
            )
        ]
    )

    # Optional RViz
    rviz_config = os.path.join(
        get_package_share_directory('realsense2_description'),
        'rviz',
        'urdf.rviz'
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config],
        condition=LaunchConfiguration('rviz')
    )

    return LaunchDescription([
        rviz_arg,
        gazebo,
        robot_state_publisher,
        spawn_entity,
        rviz
    ])
