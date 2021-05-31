from tvm import relay
from tvm.relay import op
from tvm.relay.dataflow_pattern import *


def combined_nms_pattern(boxes, scores, max_boxes_per_class, iou_threshold, score_threshold, raw_boxes, raw_scores):
    nms_out = is_op("vision.all_class_non_max_suppression")(boxes, scores, max_boxes_per_class, iou_threshold, score_threshold)
    selected_indices = is_tuple_get_item(nms_out, 0)
    num_detections = is_tuple_get_item(nms_out, 1)
    v478 = is_op("dyn.strided_slice")(selected_indices, wildcard(), num_detections, wildcard())
    v479 = is_op("dyn.strided_slice")(selected_indices, wildcard(), num_detections, wildcard())
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

    v500 = is_op("dyn.strided_slice")(selected_indices, wildcard(), num_detections, wildcard())
    v501 = is_op("transpose")(v500)
    v502 = is_op("gather_nd")(wildcard(), v501)
    v503 = is_op("shape_of")(v502)
    v504 = is_op("squeeze")(v503)
    v505 = is_op("transpose")(v499)
    v506 = is_op("arange")(wildcard(), v504, wildcard())
    v507 = is_op("transpose")(v499)
    v508 = is_op("scatter_nd")(wildcard(), v507, v502)
    v509 = is_op("topk")(v508)
    return v509


def convert_combined_nms(boxes, scores, idxs, iou_thres, num_boxes, indices):
    scores = op.expand_dims(scores, axis=-1, num_newaxis=1)
    idxs = op.expand_dims(idxs, axis=-1, num_newaxis=1)
    idxs = op.cast(idxs, "float32")
    data = op.concatenate([idxs, scores, boxes], -1)
    data = op.expand_dims(data, 0, 1)
    top_k = max_out_size = -1
    out = op.vision.non_max_suppression(
        data=data,
        valid_count=num_boxes,
        indices=indices,
        max_output_size=max_out_size,
        iou_threshold=iou_thres,
        force_suppress=False,
        top_k=top_k,
        coord_start=2,
        score_index=1,
        id_index=0,
        return_indices=True,
        invalid_to_bottom=False,
    )
    return out.tuple_value


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
            self.raw_scores
        )

    def callback(self, pre, post, node_map):
        print("matched", node_map[self.max_boxes_per_class])
        return pre
        # boxes = node_map[self.boxes][0]
        # scores = node_map[self.scores][0]
        # idxs = node_map[self.idxs][0]
        # iou_thres = node_map[self.iou_threshold][0]
        # num_boxes = node_map[self.num_boxes][0]
        # indices = node_map[self.indices][0]
        # return convert_combined_nms(boxes, scores, idxs, iou_thres, num_boxes, indices)


def rewrite_all_class_nms(mod):
    mod["main"] = rewrite(AllClassNMSRewrite(), mod["main"])
    return mod
