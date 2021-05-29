import tvm
from tvm import relay
from tvm.runtime.vm import VirtualMachine
from tvm import relay, auto_scheduler

import onnx
import onnxruntime
import numpy as np
from PIL import Image, ImageDraw, ImageColor
import matplotlib.pyplot as plt

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

def auto_schedule(mod, params, log_file):
    target = "vulkan"

    tasks, task_weights = auto_scheduler.extract_tasks(mod, params, target)

    for idx, task in enumerate(tasks):
        print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
        print(task.compute_dag)

    measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=300, timeout=50)

    # tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tuner = auto_scheduler.TaskScheduler(tasks, task_weights, load_log_file=log_file)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=20000,
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    )

    tuner.tune(tune_option)

img_path = "../77320690-099af300-6d37-11ea-9d86-24f14dc2d540.png"
image = Image.open(img_path)
image = image.resize((512, 512))
image = np.array(image)

input_data = image[np.newaxis, :]

model_path = "../models/centernet_resnet101_v1_fpn_512x512_coco17_tpu-8.onnx"
iname = "input_tensor:0"
ishape = (1, 512, 512, 3)

dtype = "uint8"
target = "vulkan"
ctx = tvm.device(target, 0)

shape_dict = {iname: ishape}
dtype_dict = {iname: dtype}

# Generate random input
np.random.seed(0)
input_dict = {iname: tvm.nd.array(input_data.astype(dtype))}
onnx_input_dict = {iname: input_data.astype(dtype)}

# ONNX runtime session
ort_sess = onnxruntime.InferenceSession(model_path)
ort_output = ort_sess.run(None, onnx_input_dict)

# TVM Compile
onnx_model = onnx.load(model_path)
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=True)
mod = relay.transform.DynamicToStatic()(mod)

log_file = "centernet_resnet101.log"

# auto_schedule(mod, params, log_file)

with auto_scheduler.ApplyHistoryBest(log_file):
    with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
        vm_exec = relay.vm.compile(mod, target, params=params)

        vm = VirtualMachine(vm_exec, ctx)

        vm.set_input("main", **input_dict)
        tvm_output = vm.run()
        tvm_output = [x.asnumpy() for x in tvm_output]

        # Check outputs
        assert(len(tvm_output)==len(ort_output))
        for i in range(len(tvm_output)):
            assert(tvm_output[i].shape == ort_output[i].shape)
            MSE = (np.square(tvm_output[i] - ort_output[i])).mean(axis=None)
            print("Mean Squared Error of output {} and shape {} is {}".format(i, tvm_output[i].shape, MSE))

        profile = False

        if profile:
            from tvm.runtime import profiler_vm
            vm = profiler_vm.VirtualMachineProfiler(vm_exec, ctx)
            report = vm.profile([input_dict[iname]], func_name="main")
            print(report)
        else:
            ftimer = vm.module.time_evaluator(
                "invoke", ctx, repeat=30, number=1
            )

            prof_res = np.array(ftimer("main").results) * 1000  # convert to millisecond
            print(prof_res)
            print("TVM VM mean inference time (std dev): %.2f ms (%.2f ms)" %
                    (np.mean(prof_res), np.std(prof_res)))
