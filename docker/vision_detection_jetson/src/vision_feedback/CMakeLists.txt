cmake_minimum_required(VERSION 3.8)
project(vision_feedback)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(ros2_interface REQUIRED)

# --- messages ---
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/BoundingBox.msg"
  "msg/BoundingBoxes.msg"
  DEPENDENCIES std_msgs
)

# --- python implementation package ---
# Installs recursively, so vision_perception/backends/ comes along with it.
ament_python_install_package(vision_perception)

# --- node entry point + operator tools ---
# build_engine.py and benchmark.py are installed too, so `ros2 run` reaches
# them from any workspace, container or not.
install(PROGRAMS
  scripts/vision_node
  scripts/build_engine.py
  scripts/benchmark.py
  DESTINATION lib/${PROJECT_NAME}
)

# --- data files ---
# `models` holds the .pt checkpoints. TensorRT engines are NOT installed here:
# they are device/TensorRT/JetPack specific and are built into the runtime
# engine cache (see vision.yaml -> engine_cache_dir).
install(DIRECTORY launch config models
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(vision_perception_tests test
    TIMEOUT 120
  )
endif()

ament_export_dependencies(rosidl_default_runtime)
ament_package()
