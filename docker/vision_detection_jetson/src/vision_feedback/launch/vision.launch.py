"""Launch the vision_feedback detection node.

All tunables live in ``config/vision.yaml`` (loaded by the node itself). This
launch file only exposes the overrides that vary per deployment. Empty string
arguments keep the YAML value, e.g.::

    ros2 launch vision_feedback vision.launch.py
    ros2 launch vision_feedback vision.launch.py backend:=torch
    ros2 launch vision_feedback vision.launch.py image_topic:=/camera/fixed/image
    ros2 launch vision_feedback vision.launch.py inference_input_shape:=576x960
    ros2 launch vision_feedback vision.launch.py inference_input_shape:=448x640
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

# Empty default => fall back to the YAML value (see config.load_params).
_STRING_ARGS = (
    "image_topic",
    "bbox_topic",
    "raw_bbox_topic",
    "device",
    "model_path",
    "backend",
    "inference_precision",
    "inference_input_shape",
    "engine_cache_dir",
)

# (name, default, python type) - always applied, never "keep the YAML value".
_TYPED_ARGS = (
    ("show_display", "true", bool),
    ("auto_build_engine", "true", bool),
    ("trt_fp16_io", "false", bool),
)


def generate_launch_description():
    declarations = [
        DeclareLaunchArgument(name, default_value="") for name in _STRING_ARGS
    ]
    declarations += [
        DeclareLaunchArgument(name, default_value=default)
        for name, default, _ in _TYPED_ARGS
    ]

    overrides = {name: LaunchConfiguration(name) for name in _STRING_ARGS}
    for name, _, value_type in _TYPED_ARGS:
        overrides[name] = ParameterValue(
            LaunchConfiguration(name), value_type=value_type)

    return LaunchDescription(declarations + [
        Node(
            package="vision_feedback",
            executable="vision_node",
            name="vision_node",
            output="screen",
            emulate_tty=True,
            parameters=[overrides],
        ),
    ])
