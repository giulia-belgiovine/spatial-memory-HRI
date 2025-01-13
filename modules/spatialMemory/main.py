import yarp
import os
import sys
import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join("../", os.path.dirname(SCRIPT_DIR)))

print(os.path.join("../../", os.path.dirname(SCRIPT_DIR)))

from motpy.core import Detection
from motpy.tracker import MultiObjectTracker
from spatialMemory.memoryManager import MemoryManager
from motpy.utils import draw_track

#############################################################
def yarpinfo(msg):
    print("\033[91m[INFO] {}\033[00m".format(msg))


def debug(msg):  # @Luca
    # print in cyan
    print("\033[96m[DEBUG] {}\033[00m".format(msg))


def warning(msg):
    print("\033[93m[WARNING] {}\033[00m".format(msg))


def error(msg):
    print("\033[91m[ERROR] {}\033[00m".format(msg))

############################################################


class SpatialMemory(yarp.RFModule):

    def __init__(self):
        yarp.RFModule.__init__(self)

        self.icub_is_moving = False
        self.module_name = None
        self.width_img = 640
        self.height_img = 480
        self.input_img_array = None
        self.mem_manager = None
        self.mot_tracker = None
        self.azimuth_head_pose = 10
        self.zone = ""
        self.role = ""
        self.zone_bin_dict = {}
        self.trackers = None

        # Memory visualization
        self.path_img_template = None
        self.cell_size = 600
        self.angle_res = 60
        self.n_mem_bin = 3  # we assume 3 memory slot
        self.memory_template = None
        self.memory_img_width = None
        self.memory_img_height = None
        self.output_frame = None
        self.output_frame_memory = None
        self.memory_display_buf_image = yarp.ImageRgb()
        self.display_buf_image = yarp.ImageRgb()

        # Ports
        self.handle_port = yarp.Port()
        self.attach(self.handle_port)

        self.input_face_port = yarp.BufferedPortBottle()
        self.input_img_port = yarp.BufferedPortImageRgb()
        self.output_img_port = yarp.Port()
        self.headpose_port = yarp.Port()
        self.gaze_events_port = yarp.BufferedPortBottle()
        self.output_trackers = yarp.Port()
        self.icub_headpose_port = yarp.Port()
        self.output_img_memory_port = yarp.Port()
        self.iKinGazeCtrl_events_port = yarp.BufferedPortBottle()
        self.input_name_port = yarp.Port()

        self.iKinGaze_rpc_port = yarp.RpcClient()
        self.iKinGaze_rpc_port.setRpcMode(True)

        self.active_trackers = []
        self.no_face_counter = 0


    def configure(self, rf):
        self.module_name = rf.check("name",
                                    yarp.Value("spatialMemory"),
                                    "module name (string)").asString()

        self.input_img_array = np.zeros((self.height_img, self.width_img, 3), dtype=np.uint8).tobytes()

        model_spec = {'order_pos': 1, 'dim_pos': 2,
                      'order_size': 0, 'dim_size': 2,
                      'q_var_pos': 1e5, 'r_var_pos': 1e-13}

        self.mot_tracker = MultiObjectTracker(dt=1/27, module_name=self.module_name, model_spec=model_spec)  # dt = 1/27 ssume 15 fps
        self.mem_manager = MemoryManager()

        # OPEN PORTS
        # Create handle port to read message
        self.handle_port.open('/' + self.module_name)

        # Create port to receive an image and face coordinates
        self.input_face_port.open('/' + self.module_name + '/face:i')
        self.input_img_port.open('/' + self.module_name + '/image:i')

        # Create a port to get information about iCub head
        self.icub_headpose_port.open('/' + self.module_name + '/icubHeadPose:i')
        self.iKinGazeCtrl_events_port.open('/' + self.module_name + '/iKinGazeCtrlEvents:i')
        # Create a client rpc port to deal with iKinGaze and transform head pixels in space coordinates
        self.iKinGaze_rpc_port.open('/' + self.module_name + '/iKinGaze:rpc')

        # Create a port to output image with trackers and memory template with labels
        self.output_img_port.open('/' + self.module_name + '/image:o')
        self.output_img_memory_port.open('/' + self.module_name + '/memory_image:o')


        # MEMORY VISUALIZATION OUTPUT
        template_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       '../../app/conf/mem_top.drawio.png')
        self.path_img_template = rf.check("img_memory_path", yarp.Value(template_folder),
                                          'Path of memory image template').asString()
        if not os.path.isfile(self.path_img_template):
            error("You should give a valid path for the memory image template!")
            return False

        self.memory_template = cv2.imread(self.path_img_template)
        self.memory_img_width = self.memory_template.shape[1]
        self.memory_img_height = self.memory_template.shape[0]
        self.output_frame_memory = np.zeros((self.memory_img_height, self.memory_img_width, 3), dtype=np.uint8)

        self.zone_bin_dict = {"left": 0, "center": 1, "right": 2}

        yarpinfo("Initialization Done. Yeah")

        return True

    def respond(self, command, reply):
        reply.clear()

        if command.get(0).asString() == "help":
            #reply.addVocab32("many")
            reply.addString("Spatial Memory module commands are:")
            reply.addString("set role<string> zone<string> -> to set the role in a zone (if there is any tracker)")
            reply.addString("get zone role<string> -> to get the zone of a specific role")
            reply.addString("accepted roles: [speaker, addressee, attendee]")
            reply.addString("accepted zones: [far_left, left, center, right, far_right")

        elif command.get(0).asString() == "clear":
            if command.get(1).asString() == "mem":
                self.mem_manager.memory = {}
                reply.addString("ok")

        elif command.get(0).asString() == "set":
            self.role = command.get(1).asString()
            self.zone = command.get(2).asString()
            print("Received command set ", command.get(1).asString(), command.get(2).asString())
            bin_idx = self.from_zone_to_bin(self.zone)
            trackers = self.mem_manager.get_trackers_by_bin(bin_idx)
            if trackers:
                for tracker in trackers:
                    tracker.status = self.role
                reply.addString("Set {} in zone {}".format(self.role, self.zone))
            else:
                reply.addString("No one in {}".format(self.zone))

        elif command.get(0).asString() == "get":
            print(command.toString())
            if command.get(1).asString() == "zone":

                self.role = command.get(2).asString()
                print("Received command get zone ", command.get(2).asString())

                positions = self.mem_manager.get_zone_by_label(self.role)
                print("positions is", positions)

                if len(positions):
                    reply.addInt8(1)
                    positions = list(set(positions))
                    position_list = reply.addList()
                    for track_zone in positions:
                        position_list.addInt8(track_zone)
                else:
                    reply.addInt8(0)

            elif command.get(1).asString() == "any":
                direction = command.get(2).asString()
                zone = command.get(3).asString()
                if (direction == "leftward" or direction == "rightward") and (zone in list(self.zone_bin_dict.keys())):
                    print(self.zone_bin_dict.keys())
                    reference_bin = list(self.zone_bin_dict.keys()).index(zone)
                    bins_list = self.mem_manager.get_occupied_bins_from_direction(direction, reference_bin)
                    reply.addInt8(len(bins_list))
                    if len(bins_list):
                        response_list = reply.addList()
                        for b in bins_list:
                            response_list.addInt8(b)
                else:
                    reply.addString("Command not valid")
            else:
                reply.addString("Command not valid")
        else:
            reply.addString("Command not valid")

        return True

    def getPeriod(self):
        """
           Module refresh rate.
           Returns : The period of the module in seconds.
        """
        return 0.5

    def updateModule(self):

        frame_input = self.get_yarp_image()
        faces_boxes = self.read_face_tracker() # faces_name
        self.check_icub_is_moving()

        if frame_input is not None:
            self.output_frame = frame_input.copy()

            if len(faces_boxes):
                detections = [Detection(box=box, feature=None) for box in faces_boxes]

                if not self.icub_is_moving:
                    # get iCub head position
                    self.azimuth_head_pose = self.get_iCub_head_position()

                    # take trackers objects that are in this position. If necessary it creates new trackers
                    trackers = self.mem_manager.get_trackers(self.azimuth_head_pose)

                    # match detections with trackers with kalman funct
                    self.active_trackers = self.mot_tracker.step(trackers, detections, self.mem_manager.memory)

                    # Update trackers to memory dictionary
                    self.mem_manager.add_trackers(self.active_trackers, self.azimuth_head_pose)
            else:
                self.no_face_counter += 1

            if self.no_face_counter == 10:
                self.mem_manager.memory = {0: None, 1: None, 2: None}
                self.no_face_counter = 0

            # Create output visualization of spatial memory
            if not self.icub_is_moving:
                if self.output_img_memory_port.getOutputCount():
                    mem_template = self.memory_template.copy()
                    self.output_frame_memory = self.draw_memory(mem_template)
                    self.write_image_memory(self.output_frame_memory)

                # Create yarp image for trackers visualization
                if self.output_img_port.getOutputCount():
                    for idx, track in enumerate(self.active_trackers):
                        draw_track(self.output_frame, track)

                    self.write_yarp_image(self.output_frame)

        # visualize with python
        # cv2.imshow("memory_view", output_frame_memory)
        # cv2.waitKey(1)

        return True

    def interruptModule(self):
        yarpinfo("Stopping the module")
        self.handle_port.interrupt()
        self.icub_headpose_port.interrupt()
        self.input_face_port.interrupt()
        self.input_img_port.interrupt()
        self.output_img_port.interrupt()
        self.headpose_port.interrupt()
        self.gaze_events_port.interrupt()
        self.output_trackers.interrupt()
        self.iKinGazeCtrl_events_port.interrupt()
        self.iKinGaze_rpc_port.interrupt()
        self.output_img_memory_port.interrupt()
        self.input_name_port.interrupt()

        return True

    def close(self):
        yarpinfo("Closing Ports")
        self.handle_port.close()
        self.icub_headpose_port.close()
        self.input_face_port.close()
        self.input_img_port.close()
        self.output_img_port.close()
        self.headpose_port.close()
        self.gaze_events_port.close()
        self.output_trackers.close()
        self.iKinGazeCtrl_events_port.close()
        self.iKinGaze_rpc_port.close()
        self.output_img_memory_port.close()
        self.input_name_port.close()

        return True

    ####################################################################################################################
    #                                            FUNCTIONS                                                             #
    ####################################################################################################################

    def get_yarp_image(self):
        '''
        Get a yarp image from robot camera and return it as a numpy array
        :return:
        '''
        frame = None
        if self.input_img_port.getInputCount():
            input_yarp_image = self.input_img_port.read(False)
            if input_yarp_image is not None:
                if input_yarp_image.width() != self.width_img or input_yarp_image.height() != self.height_img:
                    warning("input image has different size from default 640x480, we'll resize it to {}x{}".format(
                        self.width_img, self.height_img))
                    self.width_img = input_yarp_image.width()
                    self.height_img = input_yarp_image.height()
                    self.input_img_array = np.zeros((self.height_img, self.width_img, 3), dtype=np.uint8)

                # Convert yarp image to numpy array
                input_yarp_image.setExternal(self.input_img_array, self.width_img, self.height_img)
                frame = np.frombuffer(self.input_img_array, dtype=np.uint8).reshape(
                    (self.height_img, self.width_img, 3)).copy()

        return frame

    def read_face_tracker(self):
        """
        @param face_bottle: Read Yarp bottle from the ObjectDetector (Yolo) Module
        @return: list of boxes with box define as [x1,y1,x2,y2]

        """
        boxes = []
        faces = []
        face_bottle = yarp.Bottle()
        face_bottle.clear()

        # Read the face coordinates from the FaceDetector
        if self.input_face_port.getInputCount():
            face_bottle = self.input_face_port.read(False)

            if face_bottle is not None:
                for i in range(face_bottle.size()):
                    face_data = face_bottle.get(i).asList()
                    face_coordinates = face_data.get(2).asList().get(1).asList()

                    boxes.append([face_coordinates.get(0).asFloat32(), face_coordinates.get(1).asFloat32(),
                                  face_coordinates.get(2).asFloat32(), face_coordinates.get(3).asFloat32()])

                    # face_label = face_data.get(0).asList().get(1).asString()
                    # if "Recognizing" not in face_label:
                    #     if face_label == "Unknown face":
                    #         face_name = "Unknown"
                    #     else:
                    #         face_name = face_label.split("\"")[1].split(" ")[0]
                    #     faces.append(face_name)
                    #     print("face name is:", face_label)
                    #
                    #     face_score = face_data.get(1).asList().get(1).asFloat32()
                    #     print("face score is", face_score)

        return boxes  #, faces

    def write_yarp_image(self, frame):
        """
            Handle function to stream the recognize faces with their bounding rectangles
            :param img_array:
            :return:
        """
        self.display_buf_image.resize(self.width_img, self.height_img)
        self.display_buf_image.setExternal(frame.tobytes(), self.width_img, self.height_img)
        self.output_img_port.write(self.display_buf_image)

    def draw_memory(self, template_img, y_pos_increment=50):
        color_dict = {'speaker': (0, 255, 0), 'addressee': (255, 0, 0), 'person': (0, 0, 0)}

        for pos in self.mem_manager.memory.keys():
            y = 130
            list_targets = self.mem_manager.get_trackers_by_bin(pos)
            for tracker in list_targets:
                x = self.get_x_by_bin(pos)
                if tracker.id:
                    color_text = color_dict.get(tracker.status)
                    template_img = cv2.putText(template_img, str(tracker.status), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                               color_text, 2, cv2.LINE_AA)
                    # template_img = cv2.putText(template_img, str(tracker.name), (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    #                             color_text, 1, cv2.LINE_AA)
                else:
                    color = [ord(c) * ord(c) % 256 for c in tracker.id[:3]]
                    template_img = cv2.circle(template_img, (x, y), 20, color, -1)
                y -= y_pos_increment
                
        return template_img

    def get_x_by_bin5(self, pos):
        '''
        Get value of x in the memory view starting from the bin value
        :param pos: index of the bin [0, 1, 2, 3, 4]
        :return: x coordinate to display text in memory view
        '''
        bin_idx = pos
        bin_idx = abs(bin_idx-4)
        x_pos = int(bin_idx * self.cell_size) + (self.cell_size // 2)
        x_pos = x_pos + 60 # shift a bit to the left to center the text in the cell

        return x_pos

    def get_x_by_bin(self, pos):
        '''
        Get value of x in the memory view starting from the bin value
        :param pos: index of the bin [0, 1, 2, 3, 4]
        :return: x coordinate to display text in memory view
        '''
        bin_idx = pos
        bin_idx = abs(bin_idx-2)
        x_pos = int(bin_idx * self.cell_size) + (self.cell_size // 2)
        x_pos = x_pos - 50  # shift a bit to the left to center the text in the cell

        return x_pos

    def from_zone_to_bin(self, zone) -> int:
        """
        This function is to encode input from addressee estimation
        :param zone: direction from addressee
        :return: bin idx from 0 to 5 (from far left to far right)
        """
        bin_idx = self.zone_bin_dict[zone]

        return bin_idx

    def write_image_memory(self, frame):
        self.memory_display_buf_image.setExternal(frame.tobytes(), self.memory_img_width, self.memory_img_height)
        self.output_img_memory_port.write(self.memory_display_buf_image)

    def get_iCub_head_position(self):
        """
        Read the azimuth angle of the iCub head
        @return:  azimuth angle
        @rtype: int
        """
        angle = 60
        if self.icub_headpose_port.getInputCount():
            headpose_bottle = yarp.Bottle()
            headpose_bottle.clear()

            self.icub_headpose_port.read(headpose_bottle)
            if headpose_bottle is not None:
                head_position = headpose_bottle.get(0).asInt8()
                # head_position = round(headpose_bottle.get(0).asInt() / self.angle_res) * self.angle_res
                angle = head_position

        return angle

    def check_icub_is_moving(self):
        """
        Check is the iCub robot is not performing a head movements by monitoring
        ikinGazeCtrl events port
        @return: False if iCub is moving the head True otherwise
        @rtype: Bool
        """
        if self.iKinGazeCtrl_events_port.getInputCount():
            gaze_bottle = self.iKinGazeCtrl_events_port.read(False)
            if gaze_bottle is None:
                self.icub_is_moving = False
            else:
                received_event = gaze_bottle.get(0).asString()
                if received_event == 'motion-done':
                    self.icub_is_moving = False
                    #print("iCub is moving set to False")
                else:
                    self.icub_is_moving = True
                    #print("iCub is moving set to True")


if __name__ == '__main__':

    # Initialise YARP
    if not yarp.Network.checkNetwork():
        print("Unable to find a yarp server exiting ...")
        sys.exit(1)

    yarp.Network.init()
    yogaTeacherModule = SpatialMemory()

    rf = yarp.ResourceFinder()
    rf.setVerbose(True)
    # rf.setDefaultContext('yogaTeacher')
    # rf.setDefaultConfigFile('yogaTeacher.ini')

    if rf.configure(sys.argv):
        yogaTeacherModule.runModule(rf)

    sys.exit()
