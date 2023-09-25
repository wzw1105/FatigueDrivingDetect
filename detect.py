import cv2
import torch
from utils1.plots import Annotator, colors
import numpy as np

from utils1.augmentations import letterbox
from utils1.general import non_max_suppression, xyxy2xywh, scale_boxes

def detect(model, frame, stride, imgsz, enable_debug, devide = torch.device('cpu')):

    result = []
    # dataset = LoadImages(frame, img_size=imgsz, stride=stride)
    frame = np.ascontiguousarray(frame)
    img = letterbox(frame, imgsz, stride=stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img)
    img = img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    img = img.to(devide)

    #print("im shape ", img.shape)

    pred = model(img, augment=False, visualize=False)#.type(torch.FloatTensor)

    pred = non_max_suppression(pred, 0.3, 0.45, None, False, max_det=30)
    #pred = non_max_suppression(pred, 0.5, 0.45,  agnostic=False)

    # for i, det in enumerate(pred):
    #     gn = torch.tensor(frame.shape)[[1, 0, 1, 0]]
    #     if len(det):
    #         det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
    #
    #         for *xyxy, conf, cls in reversed(det):
    #             xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
    #             result.append([int(cls), xywh])
    names = model.names

    for i, det in enumerate(pred):  # per image
        det = det.cpu()
        annotator = Annotator(frame, line_width=3, example=str(names))
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                result.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), conf, c])
                if enable_debug:
                    label = f'{names[c]} {conf:.2f}' #(names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))

        # Stream results
        im0 = annotator.result()
        if enable_debug:
            cv2.imshow("result", im0)
            cv2.waitKey(50)  # 1 millisecond

    return result
