from tvm import relay
from tvm.relay import op
from tvm.relay.dataflow_pattern import *


def batched_nms_pattern(boxes, scores, idxs, iou_threshold, num_boxes, indices):
    one = is_constant()
    zero = is_constant()

    # %1824 = cast(%1823, dtype="float32");
    cast = is_op("cast")(idxs)
    mx = is_op("max")(boxes)
    add = is_op("add")(mx, one)
    mul = is_op("multiply")(cast, add)

    expand_dims = is_op("expand_dims")(mul)
    add = is_op("add")(boxes, expand_dims)

    score_expand_dims = is_op("expand_dims")(scores)

    tup = is_tuple([score_expand_dims, add])
    concat = is_op("concatenate")(tup)
    data = is_op("expand_dims")(concat)

    return is_op("vision.non_max_suppression")(
        data, num_boxes, indices, is_constant(), iou_threshold
    )


def convert_batched_nms(boxes, scores, idxs, iou_thres, num_boxes, indices):
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
        self.idxs = wildcard()
        self.iou_threshold = wildcard()
        self.num_boxes = wildcard()
        self.indices = wildcard()

        self.pattern = batched_nms_pattern(
            self.boxes,
            self.scores,
            self.idxs,
            self.iou_threshold,
            self.num_boxes,
            self.indices,
        )

    def callback(self, pre, post, node_map):
        print("matched")
        boxes = node_map[self.boxes][0]
        scores = node_map[self.scores][0]
        idxs = node_map[self.idxs][0]
        iou_thres = node_map[self.iou_threshold][0]
        num_boxes = node_map[self.num_boxes][0]
        indices = node_map[self.indices][0]
        return convert_batched_nms(boxes, scores, idxs, iou_thres, num_boxes, indices)


def rewrite_all_class_nms(mod):
    mod["main"] = rewrite(AllClassNMSRewrite(), mod["main"])
    return mod
