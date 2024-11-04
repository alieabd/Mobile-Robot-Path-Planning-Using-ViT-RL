#include <gazebo/transport/transport.hh>
#include <gazebo/msgs/msgs.hh>
#include <ros/ros.h>
#include <gazebo/gazebo_client.hh>
#include <std_msgs/String.h>

#include <iostream>

void cb(ConstContactsPtr &contactsMsg)
{
  static ros::NodeHandle n;
  static ros::Publisher pub = n.advertise<std_msgs::String>("contact_checker", 1000);

  bool ok_flag = false;

  for (int i = 0; i < contactsMsg->contact_size(); ++i) {
    gazebo::msgs::Contact contact = contactsMsg->contact(i);

    std::string collision1 = contact.collision1();
    std::string collision2 = contact.collision2();

    // check if turtlebot3_waffle contacts with any object other than ground_plane
    if (collision1.find("turtlebot3_waffle") != std::string::npos &&
        collision2 != "ground_plane::link::collision" ||
        collision2.find("turtlebot3_waffle") != std::string::npos &&
        collision1 != "ground_plane::link::collision") {
      ok_flag = true;
      break;
    }
  }
  if (ok_flag) {
    std_msgs::String msg;
    msg.data = "True";
    pub.publish(msg);
  } else {
    std_msgs::String msg;
    msg.data = "False";
    pub.publish(msg);
  }
}

int main(int _argc, char **_argv)
{
  ros::init(_argc, _argv, "gazebo_ros_node");
  ros::NodeHandle n;

  // Load gazebo
  gazebo::client::setup(_argc, _argv);

  // Create our node for communication
  gazebo::transport::NodePtr node(new gazebo::transport::Node());
  node->Init();

  // Listen to Gazebo contacts topic
  gazebo::transport::SubscriberPtr sub = node->Subscribe("/gazebo/default/physics/contacts", cb);

  ros::spin();

  // Make sure to shut everything down.
  gazebo::client::shutdown();
}
