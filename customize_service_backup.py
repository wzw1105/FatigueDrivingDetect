import math

from PIL import Image
import copy
import sys
import traceback
import os
import numpy as np
import time
import cv2
import torch
from input_reader import InputReader
from models.common import DetectMultiBackend
from tracker_xl import Tracker
import subprocess

from models.experimental import attempt_load
from detect import detect
from model_service.pytorch_model_service import PTServingBaseService

def resolve(name):
    f = os.path.join(os.path.dirname(__file__), name)
    return f

def get_model_base_path(model_dir):
    model_base_path = resolve(os.path.join("models"))
    if model_dir is None:
        if not os.path.exists(model_base_path):
            model_base_path = resolve(os.path.join("..", "models"))
    else:
        model_base_path = model_dir
    return model_base_path

def set_libs():


    #my_env = os.environ.copy()
    # append the env of target app
    python_libs_path = os.path.join(resolve("libs"))
    #print(python_libs_path)

    os.environ['LD_LIBRARY_PATH'] = f"{python_libs_path}:" + os.environ["LD_LIBRARY_PATH"]
    #os.execv(sys.argv[0], sys.argv)
    sys.path.insert(0, python_libs_path)
    #my_env["PATH"] = f"export LD_LIBRARY_PATH={python_libs_path}:\$LD_LIBRARY_PATH" + my_env["PATH"]
    #my_env["LD_LIBRARY_PATH"] = f"{python_libs_path}:" + my_env["LD_LIBRARY_PATH"]
    # update the env
    #os.environ.update(my_env)

class fatigue_driving_detection(PTServingBaseService): #PTServingBaseService
    def __init__(self, model_name, model_path): #,
        # these three parameters are no need to modify
        # super().__init__(model_name, model_path)

        self.model_name = model_name
        self.model_path = model_path
        # day_man_052_00_1 day_man_104_40_7
        # self.capture = '/home/dji/Downloads/train-data1/night_man_002_40_5.mp4' if capture is None else capture
        # self.capture = './data/night_woman_005_40_5.mp4' if capture is None else capture
        # self.capture = '/home/dji/Desktop/huawei/data/self_data/mp4s/IMG_6502.MP4'
        self.capture = 'test.mp4' #提交华为云需要用这个
        # self.capture = './data/test_mp4s/day_man_001_00_1.mp4'
        self.width, self.height = 1920, 1080
        self.cut_x1, self.cut_x2 = 800, 1920
        self.fps = 30
        self.scan_every = 1 #隔帧检测
        self.total_frames = 100000
        self.frame_3s = self.fps * 3

        self.device = torch.device('cpu')  # 大赛后台使用CPU判分
        # torchscript_model_path = os.path.join(get_model_base_path(None), "best.torchscript")
        onnx_model_path = os.path.join(get_model_base_path(None), "best.onnx")
        # openvino_model_path = os.path.join(get_model_base_path(None), "best_openvino_model")
        # libs_path = os.path.join(resolve("libs"))
        # print(libs_path)
        # subprocess.run(f'export LD_LIBRARY_PATH="{libs_path}:$LD_LIBRARY_PATH"')
        # set_libs()
        self.model = DetectMultiBackend(onnx_model_path, device=self.device, dnn=False, data='DriverFaceData.yaml', fp16=False)
        self.model.warmup(imgsz=(1, 3, 640, 640))  # warmup

        self.need_reinit = 0
        self.failures = 0
        self.enable_debug = False
        self.tracker = Tracker(self.width, self.height, scan_every=3, silent=True, enable_debug=self.enable_debug, yolo_model=self.model)

        self.loop_array_length = 5000
        self.previous_frames_result = [[0] * 5 for j in range(self.loop_array_length)]
        self.presum = [[0] * 5 for j in range(self.loop_array_length)]
        self.MAX_TOLERANCE_FRAMES = 10 #最大能容忍的误检的帧数
        self.max_tolerance_frames = [0 for _ in range(5)]
        self.enumerate_indexes = [3, 4, 2, 1]

        # self.temp = NamedTemporaryFile(delete=False)  # 用来存储视频的临时文件

    def _preprocess(self, data):
        # preprocessed_data = {}
        for k, v in data.items():
            for file_name, file_content in v.items():
                try:
                    try:
                        with open(self.capture, 'wb') as f:
                            file_content_bytes = file_content.read()
                            f.write(file_content_bytes)

                    except Exception:
                        return {"message": "There was an error loading the file"}

                    # self.capture = self.temp.name  # Pass temp.name to VideoCapture()
                except Exception:
                    return {"message": "There was an error processing the file"}
        return 'ok'

    def init(self):
        self.fps = self.input_reader.get_fps()
        self.width = self.input_reader.get_width()
        self.height = self.input_reader.get_height()
        self.total_frames = self.input_reader.get_total_frames()
        self.frame_3s = self.fps * 3
        self.scan_every = 3 * int(math.ceil((1.0 * self.fps / 10))) #设置检测的频率
        self.MAX_TOLERANCE_FRAMES = min(9, max(1, int(1.0 * self.fps / 3)))
        self.max_tolerance_frames = [self.scan_every, 3 * self.scan_every, 3 * self.scan_every, 5 * self.scan_every, 3 * self.scan_every]
        self.cut_x1, self.cut_x2 = int(5 * self.width / 12), self.width #int(7 * self.width / 24)
        self.cls_priority = [0, 1, 2, 4, 3]
        cut_image_width = self.cut_x2 - self.cut_x1
        cut_image_height = self.height
        #print("cut_image_width, cut_image_height: ", cut_image_width, cut_image_height)
        self.tracker.Init(cut_image_width, cut_image_height, self.scan_every, enable_gray=True)
        self.need_reinit = 0
        self.failures = 0

    def _inference(self, data):
        result = {"result": {"category": 0, "duration": 6000}}
        self.input_reader = InputReader(self.capture, 0, self.width, self.height, self.fps)

        #print(f"FPS: {self.fps}, Width: {self.width}, Height: {self.height}, scanf_every: {self.scan_every}, MAX_TOLERANCE: {self.MAX_TOLERANCE_FRAMES}, frame_3s: {self.frame_3s}")
        source_name = self.input_reader.name
        now = time.time()

        self.init()
        frame_index = store_index = 0
        find_maybe_result, find_final_result = False, False
        find_result_time = -1
        while self.input_reader.is_open():
            if not self.input_reader.is_open() or self.need_reinit == 1:
                self.input_reader = InputReader(self.capture, 0, self.width, self.height, self.fps, use_dshowcapture=False, dcap=None)
                self.init()

                if self.input_reader.name != source_name:
                    print(f"Failed to reinitialize camera and got {self.input_reader.name} instead of {source_name}.")
                    # sys.exit(1)
                self.need_reinit = 2
                time.sleep(0.02)
                continue
            if not self.input_reader.is_ready():
                time.sleep(0.02)
                continue

            ret, frame = self.input_reader.read()

            self.need_reinit = 0
            frame_index += 1
            if frame_index <= self.total_frames - 4 and not ret:
                #print("ERROR occurs when cur_frame_id total_frames: ", self.cur_frame_id, self.total_frames)

                continue
            #print("NORMAL cur_frame_id total_frames: ", self.cur_frame_id, self.total_frames)

            try:
                if frame is not None:
                    store_index += 1
                    frame = frame[:, self.cut_x1: self.cut_x2, :]
                    #result = detect(self.model, frame, 32, 640, self.enable_debug)
                    #self.tracker.predict(frame)
                    close_eyes, yawn, on_call, look_left_right = self.tracker.predict(frame, self.capture)
                    self.previous_frames_result[store_index][1 : 5] = int(close_eyes), int(yawn), int(on_call), int(look_left_right)

                    for cls in [3, 4, 2, 1]:
                        self.presum[store_index][cls] = self.presum[store_index - 1][cls] + self.previous_frames_result[store_index][cls]
                        cur_frame_3s_sum = self.presum[store_index][cls] - self.presum[0 if store_index - self.frame_3s <= 0 else store_index - self.frame_3s][cls]

                        if find_maybe_result:
                            if store_index - find_result_time <= self.fps:
                                if cur_frame_3s_sum > 0.85 * self.frame_3s and self.cls_priority[cls] > self.cls_priority[result['result']['category']]:
                                    #print(store_index, " change result")
                                    result['result']['category'] = cls
                            else:
                                find_final_result = True
                                break

                        if cur_frame_3s_sum > 0.85 * self.frame_3s and find_maybe_result is False:
                            #print(store_index, " find_maybe_result")
                            find_maybe_result = True
                            find_result_time = store_index
                            result['result']['category'] = cls
                    if find_final_result:
                        break

                    #print(int(close_eyes), int(yawn), int(on_call), int(look_left_right))
                    if self.enable_debug:
                        print(store_index, " on_call: ", int(on_call), " look_left_right: ", int(look_left_right), " yawn: ", int(yawn), " close_eyes: ", int(close_eyes))

                    # for behavior_index, behavior_state in zip(self.enumerate_indexes, [on_call, look_left_right, yawn, close_eyes]):
                    #     #print(store_index, " on_call: ", on_call, " look_left_right: ", look_left_right, " yawn: ", yawn, " close_eyes: ", close_eyes)
                    #     if behavior_state == True:
                    #         #print("index_ ", behavior_index)
                    #         prev_continuous_frames = 1
                    #         for i in range(max(1, store_index - self.max_tolerance_frames[behavior_index]), store_index):
                    #             try:
                    #                 #print("prev: ", self.previous_frames_result[index_][behavior_index])
                    #                 if self.previous_frames_result[i][behavior_index] != 0:
                    #                     prev_continuous_frames = max(prev_continuous_frames, self.previous_frames_result[i][behavior_index] + (store_index - i))
                    #                     #print("i ", i, ""," tolerance ", self.max_tolerance_frames[behavior_state], max(1, store_index - self.max_tolerance_frames[behavior_state] - 1),  " prev: ", self.previous_frames_result[index_][behavior_state], " add: ", store_index - i, " cur: ", prev_continuous_frames)
                    #             except Exception:
                    #                 print("Exception occurs when find previous states: ", store_index, i)
                    #                 break
                    #
                    #         #print("Current State: ", store_index, behavior_state, prev_continuous_frames)
                    #         self.previous_frames_result[store_index][behavior_index] = prev_continuous_frames
                    #         #print(f"frame_id : {self.cur_frame_id}, behavior_state: {behavior_state}")
                    #         #print(f"Info: cur_frame_id = {self.cur_frame_id}, cur_state = {state}, num_continuous_state = {prev_continuous_frames}")
                    #         #print(store_index, behavior_index, prev_continuous_frames)
                    #         # if prev_continuous_frames >= self.frame_3s - 5:
                    #         #     result['result']['category'] = behavior_index
                    #         #     find_result = True
                    #         #     break
                    #     else:
                    #         index_ = store_index % self.loop_array_length
                    #         self.previous_frames_result[index_][behavior_index] = 0
                    # if find_result:
                    #     break
                else:
                    break

            except Exception as e:
                if e.__class__ == KeyboardInterrupt:
                    print("Quitting")
                    break
                traceback.print_exc()
                self.failures += 1
                if self.failures > 30:   # 失败超过30次就默认返回
                    break
            del frame
        if self.enable_debug:
            cv2.destroyAllWindows()

        # #print(self.previous_frames_result[:store_index])
        # for j in range(1, store_index):
        #     #print(type(self.presum[j]), type(self.previous_frames_result[j]))
        #     for i in range(1, 5):
        #         self.presum[j][i] = self.presum[j - 1][i] + self.previous_frames_result[j][i]
        #
        # #print(self.presum[:store_index])
        # #result_yawn, result_close_eye, result_on_call, result_look_left_right = False, False, False, False
        # find_result = False
        # for j in [3, 4, 2, 1]:
        #     for i in range(self.frame_3s, store_index):
        #         cur_sum = self.presum[i][j] - self.presum[i - self.frame_3s][j]
        #         if cur_sum > 0.9 * self.frame_3s:
        #             # if j == 1:
        #             #     result_close_eye = True
        #             # elif j == 2:
        #             #     result_yawn = True
        #             # elif j == 3:
        #             #     result_on_call = True
        #             # else:
        #             #     result_look_left_right = True
        #             # break
        #             find_result = True
        #             result['result']['category'] = j
        #             break
        #     if find_result:
        #         break
        final_time = time.time()
        duration = int(np.round((final_time - now) * 1000))
        result['result']['duration'] = duration
        #print("RESULTS: filename = %s category = %.2f, duration = %.2f" % (source_name, result['result']['category'], result['result']['duration']))
        return result

    def _postprocess(self, data):
        os.remove(self.capture)
        return data
