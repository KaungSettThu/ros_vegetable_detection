import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import xacro

def generate_launch_description():
    # Path to your XACRO
    xacro_file = os.path.join(
        get_package_share_directory('realsense2_description'),
        'urdf',
        'test_d435i_camera.urdf.xacro'
    )

    # Process XACRO to URDF
    robot_description_config = xacro.process_file(xacro_file).toxml()

    # Gazebo server & GUI
    gazebo = Node(
        package='gazebo_ros',
        executable='gzserver',
        name='gazebo',
        output='screen'
    )

    gazebo_gui = Node(
        package='gazebo_ros',
        executable='gzclient',
        name='gazebo_gui',
        output='screen'
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description_config}]
    )

    # Spawn robot into Gazebo
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description', '-entity', 'realsense_camera'],
        output='screen'
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
        arguments=['-d', rviz_config]
    )

    return LaunchDescription([
        gazebo,
        gazebo_gui,
        robot_state_publisher,
        spawn_entity,
        rviz
    ])
