#!/usr/bin/env python3

# /* ------------- COPYRIGHT NOTICE ---------------
#
# Copyright (C) 2024 FlytBase, Inc. All Rights Reserved.
# Do not remove this copyright notice.
# Do not use, reuse, copy, merge, publish, sub-license, sell, distribute or modify this code - except without explicit,
# written permission from FlytBase, Inc.
# Contact info@flytbase.com for full license information.
# Author: Aditya Parandekar
# ------------- COPYRIGHT NOTICE ---------------*/

__copyright__ = "Copyright (C) 2024 FlytBase, Inc. All Rights Reserved. " \
                "Do not remove this copyright notice. " \
                "Do not use, reuse, copy, merge, publish, sub-license, sell, distribute " \
                "or modify this code - except without explicit, written permission from FlytBase, Inc."
__license__ = "Contact info@flytbase.com for full license information."
__author__ = "Aditya Parandekar"

import asyncio
import os
import rclpy
from rclpy.node import Node
from rcl_interfaces.srv import GetParameters
from common_interfaces.srv import ProcessImage, ToggleDetection
from common_interfaces.msg import ToggleState, Inference
from cv_bridge import CvBridge
import cv2
import threading
from libs.abstract_node_interfaces.publisher import Publisher
from libs.logging_helpers.logger_factory import LoggerFactory
from std_msgs.msg import String, Bool, Empty
from sensor_msgs.msg import Image
from libs.abstract_node_interfaces.subscriber import Subscriber
import time
from vision_stream_manager.stream_helpers.stream_lib import get_token, encode_token, is_stream_live, \
    get_token_from_token_map
from libs.abstract_node_interfaces.client import Client
import concurrent.futures

logger = LoggerFactory().get_logger().getLogger('fas.app.processor_node.main')
service_name = os.getenv('SERVICE_NAME')


class ProcessorNode(Node):
    def __init__(self, loop):
        try:
            super().__init__('processor_node')
            self.loop = loop
            self.namespace = f"{service_name}"
            self.send_goal_future = None
            self.get_result_future = None
            self.bridge = CvBridge()
            self.frame_rate = 30.0  # Target frame rate
            self.frame_duration = 1.0 / self.frame_rate  # Time per frame (in seconds)
            self.periodic_stream_alive_check_timer = None
            self.action_client = None
            self.future = None
            self.request = ProcessImage.Request()
            self.latest_frame = None
            self.detections = None
            self.mutex = True
            self.frame_counter = 0
            self.total_frames_to_skip = 2
            self.start_payload_feed_processing = False
            self.capture = None
            self.output_rtsp = None
            self.output_srt = None
            self.width = None
            self.height = None
            # TODO: Set it to false before pushing
            self.toggle_detection = False
            # TODO: REVERT TO IP 192.168.200.55 Before pushing
            self.input_stream_url = None  # Set your RTSP URL here
            # TODO: REVERT TO IP 192.168.200.55 Before pushing
            self.output_rtsp_stream_url = None

            self.output_srt_stream_url = None
            self.input_pipeline = None
            self.output_pipeline_rtsp = None
            self.output_pipeline_srt = None
            self.processing_thread = None
            self.stop_event = threading.Event()
            self.toggle_service = self.create_service(ToggleDetection, 'toggle_detection_service',
                                                      self.toggle_detection_callback)

            # self.process_detections_client = Client(node=self, dto=ProcessImage, topic='process_image_service',
            #                                         namespace=self.namespace, logger=logger,
            #                                         response_handler=self.process_detections_response_callback)

            self.toggle_detection_pub = Publisher(node=self, dto=ToggleState, topic='air/toggle_state',
                                                  namespace=self.namespace, logger=logger)
            self.process_image_pub =  Publisher(node=self, dto=Image, topic='air/process_image_req',
                                                  namespace=self.namespace, logger=logger, queue_size = 1)
            
            self.inference_sub = Subscriber(node=self, topic='air/process_image_resp', dto=Inference,
                                                           namespace=self.namespace, logger=logger,
                                                           listener_callback=self.process_detections_response_callback)

            self.start_payload_processing_sub = Subscriber(node=self, topic='air/start_payload_processing', dto=String,
                                                           namespace=self.namespace, logger=logger,
                                                           listener_callback=self.start_processing)
            self.start_health_check_pub = Publisher(node=self, topic='air/start_payload_stream_health_check', dto=String,
                                                    namespace=self.namespace, logger=logger)
            self.stop_health_check_sub = Subscriber(node=self, topic='air/stop_payload_stream_health_check', dto=Bool,
                                                    namespace=self.namespace, logger=logger,
                                                    listener_callback=self.stop_processing)
            self.param_client = Client(node=self, dto=GetParameters, topic='/parameter_server/get_parameters',
                                       namespace=self.namespace, logger=logger)

            self.heartbeat_pub = Publisher(node=self, dto=Empty, topic='/processor_node/heartbeat',
                                                  namespace=self.namespace, logger=logger)

            self.heartbeat_timer = self.create_timer(5.0, self.publish_heartbeat)

            msg = ToggleState()
            msg.status = self.toggle_detection
            self.toggle_detection_pub.publish(msg=msg)

            logger.info(f"\n\n{[self.namespace]} Successfully initialized processor node")

            asyncio.run_coroutine_threadsafe(self.frame_processing_loop(), self.loop)
        except Exception as e:
            logger.exception(f"While initializing processor node the following exception occurred {e}")


    def publish_heartbeat(self):
        # Publish a heartbeat message to indicate the node is still active
        heartbeat_msg = Empty()
        self.heartbeat_pub.publish(heartbeat_msg)
        # self.get_logger().info('Published heartbeat')  # Optional logging

    def release_resources(self, output_only=False, input_only=False):
        try:
            if self.capture is not None and not output_only:
                self.capture.release()
            if self.output_rtsp is not None and not input_only:
                self.output_rtsp.release()
            if self.output_srt is not None and not input_only:
                self.output_srt.release()
        except Exception as e:
            logger.exception(f"While releasing resources due to exception {e}")

    def update_pipeline_tokens(self, stream_name, token):
        self.input_stream_url = f"rtsp://192.168.200.55:8554/{stream_name}_raw_stream"
        self.output_rtsp_stream_url = f"rtsp://192.168.200.55:8554/{stream_name}"
        self.output_srt_stream_url = f'srt://srt-auto.millicast.com:10000?streamid={stream_name}%3Ft%3D{token}'
        self.input_pipeline = (
            f"rtspsrc location={self.input_stream_url} buffer-mode=none latency=0 tcp-timeout=5000000 ! "
            "rtph264depay ! h264parse ! nvh264dec ! videoconvert ! appsink wait-on-eos=false drop=true"
        )
        self.output_pipeline_rtsp = (
            "appsrc max-bytes=500000 max-latency=0 ! videoconvert ! video/x-raw,format=I420 ! videoscale ! "
            "queue max-size-buffers=0 flush-on-eos=true ! "
            "nvh264enc ! "
            f"rtspclientsink location={self.output_rtsp_stream_url} latency=0"
        )

        self.output_pipeline_srt = (
            "appsrc max-bytes=500000 max-latency=0 ! videoconvert ! video/x-raw,format=I420 ! videoscale ! "
            "queue max-size-buffers=0 flush-on-eos=true max-size-bytes=500000 ! "
            "nvh264enc preset=4 bitrate=3000 rc-mode=3 zerolatency=true ! "
            "mpegtsmux ! "
            f"srtsink uri={self.output_srt_stream_url} latency=0"

        )

    def start_processing(self, msg):
        asyncio.run_coroutine_threadsafe(self.get_live_status_and_start_processing(msg), self.loop)

    def stop_processing(self, msg):
        self.release_resources()
        self.start_payload_feed_processing = False

    async def get_live_status_and_start_processing(self, msg):
        try:
            # Release any previous resources
            self.release_resources()

            logger.info(f"Fetching camera tokens with param stream_name_token_map")
            request = GetParameters.Request()
            stream_name = msg.data
            request.names = ["associated_device_params/stream_name_token_map"]
            result = self.param_client.send_request(request, sync=True)

            if result and result.values:
                param = result.values[0]
                stream_name_token_map = param.string_value
                logger.info(
                    f"Got global param 'stream_name_token_map': {stream_name_token_map} for stream name {stream_name}")

                token = get_token_from_token_map(stream_name, stream_name_token_map)
                encoded_token = encode_token(token)
                self.update_pipeline_tokens(stream_name, encoded_token)

                logger.info(f"Recreating new output pipelines for pushing to SRT {self.output_srt_stream_url}")

                if not await is_stream_live(self.input_stream_url):
                    logger.warn('Failed to get Raw Payload feed from video server, Retry by starting it again')
                else:
                    logger.info(f'Received payload raw stream {self.input_stream_url}')
                    msg = String()
                    msg.data = stream_name

                    # Stop the previous thread if it's running
                    self.stop_previous_thread()

                    # Start a new thread for video capture
                    self.processing_thread = threading.Thread(target=self.init_pipeline_objects, args=(msg,), daemon=True)
                    self.stop_event.clear()  # Clear the stop flag for the new thread
                    self.processing_thread.start()
            else:
                logger.info(f"No results found while attempting to stream")
        except Exception as e:
            logger.exception(f"Failed to start payload processing due to exception {e}")

    def stop_previous_thread(self):
        try:

            """Stop the currently running thread by signaling the stop event."""
            if self.processing_thread and self.processing_thread.is_alive():
                logger.info("Stopping the previous video processing thread.")
                self.stop_event.set()  # Signal the thread to stop
                self.processing_thread.join()  # Wait for the thread to finish
            else:
                logger.info("No thread to stop and join")

        except Exception as e:
            logger.exception(f"Failed to delete thread due to: {e}")


    def init_pipeline_objects(self, msg):
        """This method runs in a separate thread to avoid blocking."""
        try:

            logger.info("Starting video processing in a new thread.")
            self.capture = cv2.VideoCapture(self.input_pipeline, cv2.CAP_GSTREAMER)
            self.output_rtsp = cv2.VideoWriter(self.output_pipeline_rtsp, cv2.CAP_GSTREAMER,
                                                0, self.frame_rate,
                                                (self.width, self.height), True)
            self.output_srt = cv2.VideoWriter(self.output_pipeline_srt, cv2.CAP_GSTREAMER,
                                                0, self.frame_rate,
                                                (self.width, self.height), True)
            self.start_payload_feed_processing = True
            self.start_health_check_pub.publish(msg=msg)

        except Exception as e:
            logger.exception(f"Failed to initialize pipeline objects due to : {e}")


    async def frame_processing_loop(self):
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

        while rclpy.ok():
            try:
                start_time = time.time()  # Track the start time of the loop

                if self.start_payload_feed_processing:
                    try:
                        # Run capture.read() in a separate thread with a timeout
                        ret, cv_image = await asyncio.wait_for(
                            asyncio.get_event_loop().run_in_executor(executor, self.capture.read),
                            timeout=0.5  # Timeout in seconds
                        )
                    except asyncio.TimeoutError:
                        logger.warn("Frame capture timed out, skipping frame")
                        self.release_resources(input_only=True)
                        self.capture = cv2.VideoCapture(self.input_pipeline, cv2.CAP_GSTREAMER)
                        await asyncio.sleep(0.25)
                        continue  # Skip further processing for this loop iteration

                    if not ret:
                        logger.warn(f"No valid frames found")
                        self.release_resources(input_only=True)
                        self.capture = cv2.VideoCapture(self.input_pipeline, cv2.CAP_GSTREAMER)
                        await asyncio.sleep(0.25)
                        continue

                    try:
                        height, width, _ = cv_image.shape
                    except AttributeError:
                        logger.error("Failed to get frame shape, skipping frame")
                        continue

                    # Check if video properties have changed or the output needs to be cleared
                    if (height != self.height or width != self.width or self.output_rtsp is None or
                            self.output_srt is None):
                        self.release_resources(output_only=True)

                        self.width = width
                        self.height = height

                        self.output_rtsp = cv2.VideoWriter(self.output_pipeline_rtsp, cv2.CAP_GSTREAMER,
                                                           0, self.frame_rate,
                                                           (self.width, self.height), True)
                        self.output_srt = cv2.VideoWriter(self.output_pipeline_srt, cv2.CAP_GSTREAMER, 0,
                                                          self.frame_rate,
                                                          (self.width, self.height), True)
                        logger.info(
                            f"Restarting output pipelines due to resolution changes or unavailability of resources")

                    # Process the frame if detection is enabled
                    # logger.debug(f"Mutex is set as: {self.mutex}")
                    if self.mutex and self.toggle_detection:
                        if self.frame_counter % self.total_frames_to_skip == 0:
                            self.process_frame(cv_image)
                            
                        self.frame_counter += 1

                        

                    # Annotate the frame if detections are available
                    if self.detections is not None:
                        labels, bbox, track_ids = (self.detections.labels, self.detections.bboxes,
                                                   self.detections.track_ids)
                        cv_image = self.annotate_frame(cv_image, labels, bbox, track_ids)

                    # Write the processed frame to the output
                    self.output_srt.write(cv_image)
                    self.output_rtsp.write(cv_image)

                # Calculate how long the frame processing took
                elapsed_time = time.time() - start_time

                # Sleep for the remainder of the frame duration to maintain 30 FPS
                sleep_time = self.frame_duration - elapsed_time
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

            except Exception as e:
                logger.error(f'While processing frames the following exception occurred: {e}')
                self.release_resources()

    def annotate_frame(self, frame, labels, bboxes, track_ids):
        try:
            if not self.toggle_detection:
                return frame

            for i, label in enumerate(labels):
                track_id = track_ids[i]
                if track_id != -1:
                    label = label + "  " + str(track_id)
                x1 = int(bboxes[4 * i])
                y1 = int(bboxes[4 * i + 1])
                x2 = int(bboxes[4 * i + 2])
                y2 = int(bboxes[4 * i + 3])

                color = (0, 255, 0)
                thickness = 2

                # Draw the bounding box
                frame[y1:y1 + thickness, x1:x2] = color
                frame[y2:y2 + thickness, x1:x2] = color
                frame[y1:y2, x1:x1 + thickness] = color
                frame[y1:y2, x2:x2 + thickness] = color

                # Annotate the label
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                text_size = cv2.getTextSize(label, font, font_scale, 1)[0]

                text_x = x1
                text_y = y1 - 10 if y1 - 10 > 10 else y1 + 10 + text_size[1]

                # Draw the background for text
                cv2.rectangle(frame, (text_x, text_y - text_size[1]), (text_x + text_size[0], text_y + 5), color, -1)

                # Put the label text
                cv2.putText(frame, label, (text_x, text_y), font, font_scale, (0, 0, 0), 1, cv2.LINE_AA)

            return frame
        except Exception as e:
            logger.error(f"Exception while annotating frame: {e}")
            return frame

    def process_frame(self, frame):
        try:
            self.mutex = False
            converted_image = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            self.process_image_pub.publish(converted_image)

        except Exception as e:
            logger.error(f"Exception in process_frame: {e}")

    def process_detections_response_callback(self, msg):
        try:
            self.detections = msg
            self.mutex = True

        except Exception as e:
            logger.error(f"Error occurred while processing detections response_callback: {e}")

    def toggle_detection_callback(self, request, response):
        logger.info(f"Toggle detection request received {request.enable_detection}")
        self.toggle_detection = request.enable_detection
        msg = ToggleState()
        msg.status = self.toggle_detection
        self.toggle_detection_pub.publish(msg=msg)
        if not self.toggle_detection:
            self.detections = None
            self.mutex = True
        logger.info(f"\n\n\n Detection toggled to: {'enabled' if self.toggle_detection else 'disabled'}\n\n\n")
        response.result = 0
        response.job_id = request.job_id
        return response


def spin_rclpy_node(node):
    try:
        rclpy.spin(node)  # Block and process ROS 2 events
    except Exception as e:
        node.get_logger().error(f"Exception in spin loop: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()


async def new():
    rclpy.init(args=None)

    # Create the ROS 2 node
    node = ProcessorNode(loop=asyncio.get_event_loop())

    # Create and start a thread to spin the ROS 2 node
    spin_thread = threading.Thread(target=spin_rclpy_node, args=(node,))
    spin_thread.start()

    try:
        # Run asyncio tasks here
        while rclpy.ok():
            await asyncio.sleep(2)  # Adjust sleep time as necessary
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join()  # Ensure the thread completes before exiting


def main(args=None):
    # Initialize ROS 2 client library
    asyncio.run(new())


if __name__ == "__main__":
    main()  # This should run the coroutine properly
