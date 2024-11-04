#!/usr/bin/env python3

import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from time import sleep
from std_msgs.msg import String




class MarkerBasics(object):

    def __init__(self,x,y):
        self.marker_objectlisher = rospy.Publisher('/marker_basic', Marker, queue_size=1)
        self.rate = rospy.Rate(1)
        self.x = x
        self.y = y
        self.init_marker(self.x,self.y,index=0,z_val=0)
        
    
    def init_marker(self,x,y,index=0, z_val=0):
        self.x = x
        self.y = y
        self.marker_object = Marker()
        self.marker_object.header.frame_id = "odom"
        self.marker_object.header.stamp    = rospy.get_rostime()
        self.marker_object.ns = "haro"
        self.marker_object.id = index
        self.marker_object.type = Marker.SPHERE
        self.marker_object.action = Marker.ADD
        
        my_point = Point()
        my_point.z = z_val
        my_point.x = self.x
        my_point.y = self.y
        self.marker_object.pose.position = my_point
        
        self.marker_object.pose.orientation.x = 0
        self.marker_object.pose.orientation.y = 0
        self.marker_object.pose.orientation.z = 0.0
        self.marker_object.pose.orientation.w = 1.0
        self.marker_object.scale.x = 0.1
        self.marker_object.scale.y = 0.1
        self.marker_object.scale.z = 1.0
    
        self.marker_object.color.r = 0.0
        self.marker_object.color.g = 0.0
        self.marker_object.color.b = 1.0
        # This has to be, otherwise it will be transparent
        self.marker_object.color.a = 1.0
            
        # If we want it for ever, 0, otherwise seconds before desapearing
        self.marker_object.lifetime = rospy.Duration(0)
    
    def start(self):
        #while not rospy.is_shutdown():
        #never stops
        self.marker_objectlisher.publish(self.marker_object)
        self.rate.sleep()
   




def callback(data):
    x,y = [float(a) for a in data.data.split(' ')]
    print(f'x ===> {x}')
    print(f'y ===> {y}')
    markerbasics_object = MarkerBasics(x,y)
    markerbasics_object.start()




def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.

    rospy.Subscriber("chatter", String, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()




if __name__ == '__main__':
    rospy.init_node('marker_basic_node', anonymous=True)
    listener()
    #for x_data,y_data in zip([0.5,1,1.5,2,2.5,3,3.5,4],[0.5,1,1.5,2,2.5,3,3.5,4]):
    #    markerbasics_object = MarkerBasics(x_data,y_data)
    #    markerbasics_object.start()
    #    sleep(3)
    