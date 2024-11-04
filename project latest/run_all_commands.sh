#!/bin/bash

# Create a new Tmux session
tmux new-session -d -s my_session

# Split the window into five panes
tmux split-window -h
tmux split-window -v
tmux split-window -v
tmux split-window -v


# Set TURTLEBOT3_MODEL in each pane
tmux send-keys -t my_session:0.0 'cd project/third_try' Enter
tmux send-keys -t my_session:0.1 'export TURTLEBOT3_MODEL=waffle' Enter
tmux send-keys -t my_session:0.2 'export TURTLEBOT3_MODEL=waffle' Enter
tmux send-keys -t my_session:0.3 'export TURTLEBOT3_MODEL=waffle' Enter
tmux send-keys -t my_session:0.4 'cd listener/build/' Enter

# Run commands in each pane
tmux send-keys -t my_session:0.0 'sleep 6 && rosrun my_robot_controller vit_modified_RL.py' Enter
tmux send-keys -t my_session:0.1 'sleep 4 && roslaunch turtlebot3_gazebo turtlebot3_gazebo_rviz.launch' Enter
tmux send-keys -t my_session:0.2 'sleep 4 && rosrun my_robot_controller basic_marker.py' Enter
tmux send-keys -t my_session:0.3 'roslaunch turtlebot3_gazebo reza.launch' Enter
tmux send-keys -t my_session:0.4 'sleep 5 && ./listener' Enter

# Attach to the Tmux session
tmux attach-session -t my_session

