import tvm
from tvm import relay
from tvm.runtime.vm import VirtualMachine

import onnx
import onnxruntime
import numpy as np
from PIL import Image, ImageDraw, ImageColor
import matplotlib.pyplot as plt

from rewrite_combined_nms import rewrite_all_class_nms


# COCO categories
category_map = {
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    11: 'fire hydrant',
    13: 'stop sign',
    14: 'parking meter',
    15: 'bench',
    16: 'bird',
    17: 'cat',
    18: 'dog',
    19: 'horse',
    20: 'sheep',
    21: 'cow',
    22: 'elephant',
    23: 'bear',
    24: 'zebra',
    25: 'giraffe',
    27: 'backpack',
    28: 'umbrella',
    31: 'handbag',
    32: 'tie',
    33: 'suitcase',
    34: 'frisbee',
    35: 'skis',
    36: 'snowboard',
    37: 'sports ball',
    38: 'kite',
    39: 'baseball bat',
    40: 'baseball glove',
    41: 'skateboard',
    42: 'surfboard',
    43: 'tennis racket',
    44: 'bottle',
    46: 'wine glass',
    47: 'cup',
    48: 'fork',
    49: 'knife',
    50: 'spoon',
    51: 'bowl',
    52: 'banana',
    53: 'apple',
    54: 'sandwich',
    55: 'orange',
    56: 'broccoli',
    57: 'carrot',
    58: 'hot dog',
    59: 'pizza',
    60: 'donut',
    61: 'cake',
    62: 'chair',
    63: 'couch',
    64: 'potted plant',
    65: 'bed',
    67: 'dining table',
    70: 'toilet',
    72: 'tv',
    73: 'laptop',
    74: 'mouse',
    75: 'remote',
    76: 'keyboard',
    77: 'cell phone',
    78: 'microwave',
    79: 'oven',
    80: 'toaster',
    81: 'sink',
    82: 'refrigerator',
    84: 'book',
    85: 'clock',
    86: 'vase',
    87: 'scissors',
    88: 'teddy bear',
    89: 'hair drier',
    90: 'toothbrush'
}

def draw_detection(draw, d, c):
    width, height = draw.im.size
    # the box is relative to the image size so we multiply with height and width to get pixels
    top = d[0] * height
    left = d[1] * width
    bottom = d[2] * height
    right = d[3] * width
    top = max(0, np.floor(top + 0.5).astype('int32'))
    left = max(0, np.floor(left + 0.5).astype('int32'))
    bottom = min(height, np.floor(bottom + 0.5).astype('int32'))
    right = min(width, np.floor(right + 0.5).astype('int32'))
    label = category_map[c]
    label_size = draw.textsize(label)
    if top - label_size[1] >= 0:
        text_origin = tuple(np.array([left, top - label_size[1]]))
    else:
        text_origin = tuple(np.array([left, top + 1]))
    color = ImageColor.getrgb("red")
    thickness = 0
    draw.rectangle([left + thickness, top + thickness, right - thickness, bottom - thickness],
    outline=color)
    draw.text(text_origin, label, fill=color)

img_path = "../77320690-099af300-6d37-11ea-9d86-24f14dc2d540.png"
image = Image.open(img_path)
image = image.resize((512, 512))
image = np.array(image)

input_data = image[np.newaxis, :]

model_path = "../models/centernet_resnet101_v1_fpn_512x512_coco17_tpu-8.onnx"
iname = "input_tensor:0"
ishape = (1, 512, 512, 3)

mod_layout = "NHWC"
dtype = "uint8"
target = "vulkan -from_device=0"
target = "llvm"

dev = tvm.device(target, 0)
shape_dict = {iname: ishape}
dtype_dict = {iname: dtype}

# Generate random input
np.random.seed(0)
input_dict = {iname: tvm.nd.array(input_data.astype(dtype))}
onnx_input_dict = {iname: input_data.astype(dtype)}

onnx_model = onnx.load(model_path)

# ONNX runtime session
ort_sess = onnxruntime.InferenceSession(model_path)
ort_output = ort_sess.run(None, onnx_input_dict)

# TVM Compile

mod, params = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=True)
mod = relay.transform.DynamicToStatic()(mod)

from tvm.relay.transform import InferType, ToMixedPrecision, mixed_precision
mod = ToMixedPrecision("float16")(mod)

print(len(ort_output))
do_write = True

if do_write:
    mod = rewrite_all_class_nms(mod)
    with tvm.transform.PassContext(opt_level=3):
        opt_mod, _ = relay.optimize(mod, target=target, params=params)
        print(opt_mod)
        json, lib, params = relay.build(mod, target=target, params=params)


    # print(relay.transform.InferType()(mod))
    ctx = tvm.device(target, 0)
    runtime = tvm.contrib.graph_executor.create(json, lib, ctx)
    runtime.set_input(**params)
    runtime.set_input(iname, input_dict[iname])
    runtime.run()
    tvm_output = [runtime.get_output(i).numpy() for i in range(4)]
else:
    # Compile with VM
    with tvm.transform.PassContext(opt_level=3):
        vm_exec = relay.vm.compile(mod, target, params=params)

    # from tvm.runtime import profiler_vm
    # vm = profiler_vm.VirtualMachineProfiler(vm_exec, dev)

    # report = vm.profile(input_dict[iname], func_name="main")
    # print(report)

    vm = VirtualMachine(vm_exec, dev)

    # Run inference on sample data with TVM
    vm.set_input("main", **input_dict)  # required
    tvm_output = vm.run()
    tvm_output = [x.asnumpy() for x in tvm_output]
    # ftimer = vm.module.time_evaluator(
    #     "invoke", tvm.cpu(0), repeat=3, number=3
    # )
    # prof_res = np.array(ftimer("main").results) * 1000  # convert to millisecond
    # print("TVM VM mean inference time (std dev): %.2f ms (%.2f ms)" %
    #         (np.mean(prof_res), np.std(prof_res)))

# Check outputs
assert(len(tvm_output)==len(ort_output))
for i in range(len(tvm_output)):
    assert(tvm_output[i].shape == ort_output[i].shape)
    MSE = (np.square(tvm_output[i] - ort_output[i])).mean(axis=None)
    print ("Mean Squared Error of output {} and shape {} is {}".format(i, tvm_output[i].shape, MSE))

# # Produce output
detection_boxes = tvm_output[0][0]
detection_scores = tvm_output[2][0]
detection_classes = tvm_output[1][0]

img = Image.open(img_path)
draw = ImageDraw.Draw(img)
for detection in range(len(detection_boxes)):
    if detection_scores[detection] >= 0.2: # hardcoded threshold
        c = detection_classes[detection] + 1
        d = detection_boxes[detection]
        draw_detection(draw, d, c)

# Write detection outptu to output.pdf
plt.figure(figsize=(80, 40))
plt.axis('off')
plt.imshow(img)
plt.savefig('output.pdf')
print("Wrote output of object detection to output.pdf")
