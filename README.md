This repository provides an overview of a robotics project that explores path planning techniques using deep learning and reinforcement learning.

the original article : https://ieeexplore.ieee.org/abstract/document/10412450


<img width="1115" alt="image" src="https://github.com/user-attachments/assets/c33531f9-7adb-4fc0-b737-63424e008465">






## Setup Instructions

### Prerequisites

- ROS Noetic
- Python 3
- Gazebo
- Catkin

### Building the Workspace

1. Navigate to the `catkin_ws` directory:
    ```sh
    cd catkin_ws
    ```

2. Build the workspace:
    ```sh
    catkin_make
    ```

3. Source the setup file:
    ```sh
    source devel/setup.bash
    ```

### Running the Project

1. Launch the Gazebo simulation:
    ```sh
    export TURTLEBOT3_MODEL=waffle
    roslaunch turtlebot3_gazebo turtlebot3_gazebo_rviz.launch
    ```

2. Run the robot controller:
    ```sh
    rosrun my_robot_controller basic_marker.py
    ```

3. Run the ViT & RL model:
    ```sh
    rosrun my_robot_controller vit_modified_RL.py
    ```

4. Run the contact publisher:
    ```sh
    ./listener
    ```
    

### Notebooks

The project includes several Jupyter notebooks for data processing and analysis:

To run the notebooks, navigate to the `project latest` directory and start Jupyter:

```sh
cd project latest
jupyter notebook
