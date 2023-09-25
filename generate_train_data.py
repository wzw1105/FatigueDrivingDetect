from models.common import DetectMultiBackend
from utils1.labelFile import *
from tracker_xl import *
import os
import cv2


def format_shape(s):
    return dict(label=s.label,
                points=[(p.x(), p.y()) for p in s.points],
                # add chris
                difficult=False,
                direction=s.direction,
                center=s.center,
                isRotated=s.isRotated)

a = LabelFile()
data_dir = '/home/dji/Downloads/data_split_1_CHECKED'

device = torch.device('cuda:0')
model = DetectMultiBackend('/home/dji/PycharmProjects/CloudFaceYolo/model/best.pt', device=device, dnn=False, data=None, fp16=False)
tracker = Tracker(0, 0, scan_every=1, silent=True, enable_debug=False, yolo_model=model)
img_names = [x for x in os.listdir(data_dir) if x.endswith('.jpg')]
predef_label = ['postive_face', 'negative_face', 'open_eye', 'closed_eye', 'open_mouth', 'closed_mouth', 'phone_and_hand']

extracted_type = [0 for _ in range(7)]
tot_imgs = 0
for img_name in img_names:
    img_path = os.path.join(data_dir, img_name)
    xml_path = os.path.join(data_dir, os.path.splitext(img_name)[0] + '.xml')

    img = cv2.imread(img_path)
    height, width, _ = img.shape
    tracker.Init(width, height, 1, False)
    _, result = tracker.predict(img, device=device)


    #print(cls, result)
    # shape = [dict(label=,
    #               points=boxes,
    #               # add chris
    #               difficult=False,
    #               direction=0,
    #               center=0,
    #               isRotated=0)]
    result = sorted(result, key=lambda x:x[5])
    shape = []
    for x1, y1, x2, y2, conf, cls in result:
        extracted_type[cls] += 1
        shape.append(dict(label=predef_label[cls],
                      points=[(x1, y1), (x2, y2)],
                      # add chris
                      difficult=False,
                      direction=0,
                      center=0,
                      isRotated=0))

    a.savePascalVocFormat(xml_path, shape, img_path)
    tot_imgs += 1
    print(f"Saved label to {xml_path}")


for cls, num in zip(predef_label, extracted_type):
    print(f"{cls}: {num},", end=' ')
print('\n')
print(f"Generate {tot_imgs} instances totally.")