from tvm.relay import op as _op
from tvm.relay import expr as _expr
from tvm.relay.frontend.tensorflow import convert_combined_nms_with_all_class_nms
from tvm.relay.dataflow_pattern import *


def combined_nms_pattern(
    boxes,
    scores,
    max_boxes_per_class,
    iou_threshold,
    score_threshold,
    raw_boxes,
    raw_scores,
):
    nms_out = is_op("vision.all_class_non_max_suppression")(
        boxes, scores, max_boxes_per_class, iou_threshold, score_threshold
    )
    selected_indices = is_tuple_get_item(nms_out, 0)
    num_detections = is_tuple_get_item(nms_out, 1)
    v478 = is_op("dyn.strided_slice")(
        selected_indices, wildcard(), num_detections, wildcard()
    )
    v479 = is_op("dyn.strided_slice")(
        selected_indices, wildcard(), num_detections, wildcard()
    )
    v480 = is_op("strided_slice")(v478)
    v481 = is_op("strided_slice")(v479)
    v482 = is_tuple([v480, v481])
    v483 = is_op("concatenate")(v482)
    v484 = is_op("transpose")(v483)
    v485 = is_op("gather_nd")(wildcard(), v484)
    v486 = is_op("squeeze")(v485)
    v487 = is_op("maximum")(v486, wildcard())
    v488 = is_op("minimum")(v487, wildcard())
    v489 = is_op("squeeze")(v480)
    v490 = is_op("less")(v489, wildcard())
    v491 = is_op("add")(v489, wildcard())
    v492 = is_op("where")(v490, v491, v489)
    v493 = is_op("take")(wildcard(), v492)
    v494 = is_op("cumsum")(v493)
    v495 = is_op("transpose")(v480)
    v496 = is_op("gather_nd")(v494, v495)
    v497 = is_op("expand_dims")(v496)
    v498 = is_tuple([v480, v497])
    v499 = is_op("concatenate")(v498)
    v500 = is_op("dyn.strided_slice")(
        selected_indices, wildcard(), num_detections, wildcard()
    )
    v501 = is_op("transpose")(v500)
    v502 = is_op("gather_nd")(wildcard(), v501)
    v503 = is_op("shape_of")(v502)
    v504 = is_op("squeeze")(v503)
    v505 = is_op("transpose")(v499)
    v506 = is_op("arange")(wildcard(), v504, wildcard())
    v507 = is_op("transpose")(v499)
    v508 = is_op("scatter_nd")(wildcard(), v507, v502)
    v509 = is_op("topk")(v508)
    v510 = is_tuple_get_item(v509, 1)
    v511 = is_op("less")(v510, wildcard())
    v512 = is_op("add")(v510, wildcard())
    v513 = is_op("scatter_nd")(wildcard(), v505, v506)
    v514 = is_op("where")(v511, v512, v510)
    v515 = is_op("gather")(v513, v514)
    v516 = is_op("nn.pad")(v515, wildcard())
    v517 = is_op("add")(v516, wildcard())
    v518 = is_op("nn.pad")(v488, wildcard())
    v519 = is_op("shape_of")(v518)
    v520 = is_op("take")(v519, wildcard())
    v521 = is_op("less")(v517, wildcard())
    v522 = is_op("add")(v517, v520)
    v523 = is_op("where")(v521, v522, v517)
    v524 = is_op("dyn.strided_slice")(
        selected_indices, wildcard(), num_detections, wildcard()
    )
    v525 = is_op("strided_slice")(v524)
    v526 = is_op("squeeze")(v525)
    v527 = is_op("nn.pad")(v526, wildcard())
    v528 = is_op("shape_of")(v527)
    v529 = is_op("take")(v528, wildcard())
    v530 = is_op("less")(v517, wildcard())
    v531 = is_op("add")(v517, v529)
    v532 = is_op("where")(v530, v531, v517)
    v533 = is_op("take")(v527, v532)
    v534 = is_op("cast")(v533)
    v535 = is_tuple_get_item(v509, 0)
    v536 = is_op("greater")(v515, wildcard())
    v537 = is_op("cast")(v536)
    v538 = is_op("sum")(v537)
    v539 = is_op("take")(v518, v523)
    v540 = is_op("add")(v534, wildcard())
    v541 = is_op("nn.pad")(v535, wildcard())
    v542 = is_op("cast")(v538)
    return is_tuple([v539, v540, v541, v542, raw_boxes, raw_scores])


def convert_combined_nms(
    batch_size,
    max_output_boxes_per_batch,
    num_class,
    boxes,
    scores,
    max_output_boxes_per_class,
    iou_threshold,
    score_threshold,
    max_total_size,
    clip_boxes,
    raw_boxes,
    raw_scores,
):
    (
        nmsed_boxes,
        nmsed_scores,
        nmsed_classes,
        num_detections,
    ) = convert_combined_nms_with_all_class_nms(
        batch_size,
        max_output_boxes_per_batch,
        num_class,
        boxes,
        scores,
        max_output_boxes_per_class,
        iou_threshold,
        score_threshold,
        max_total_size,
        clip_boxes,
    )

    return _expr.Tuple(
        [
            nmsed_boxes,
            _op.add(nmsed_classes, _op.const(1, "float32")),
            nmsed_scores,
            _op.cast(num_detections, "float32"),
            raw_boxes,
            raw_scores,
        ]
    )


class AllClassNMSRewrite(DFPatternCallback):
    def __init__(self):
        super().__init__()
        # exprs I want to extract
        self.boxes = wildcard()
        self.scores = wildcard()
        self.max_boxes_per_class = wildcard()
        self.iou_threshold = wildcard()
        self.score_threshold = wildcard()
        self.raw_boxes = wildcard()
        self.raw_scores = wildcard()

        self.pattern = combined_nms_pattern(
            self.boxes,
            self.scores,
            self.max_boxes_per_class,
            self.iou_threshold,
            self.score_threshold,
            self.raw_boxes,
            self.raw_scores,
        )

    def callback(self, pre, post, node_map):
        boxes = node_map[self.boxes][0]
        scores = node_map[self.scores][0]
        max_boxes_per_class = node_map[self.max_boxes_per_class][0]
        iou_thres = node_map[self.iou_threshold][0]
        score_thres = node_map[self.score_threshold][0]
        raw_boxes = node_map[self.raw_boxes][0]
        raw_scores = node_map[self.raw_scores][0]
        max_output_boxes_per_batch = 90 * 12804  # TODO
        return convert_combined_nms(
            1,
            max_output_boxes_per_batch,
            90,
            boxes,
            scores,
            max_boxes_per_class.data.numpy().item(),
            iou_thres,
            score_thres,
            100,
            True,
            raw_boxes,
            raw_scores,
        )


def rewrite_all_class_nms(mod):
    mod["main"] = rewrite(AllClassNMSRewrite(), mod["main"])
    return mod
