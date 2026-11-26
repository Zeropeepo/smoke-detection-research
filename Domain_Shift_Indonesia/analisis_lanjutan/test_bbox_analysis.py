import tempfile
import unittest
from pathlib import Path

from bbox_analysis import audit_case, box_iou, fn_indicators, read_boxes


def box(class_id, xywh, confidence=None):
    return {"class_id": class_id, "xywh": xywh, "confidence": confidence}


class MatchingTests(unittest.TestCase):
    def test_one_ground_truth_cannot_match_two_predictions(self):
        gt = [box(0, (0.5, 0.5, 0.2, 0.2))]
        pred = [box(0, gt[0]["xywh"], 0.9), box(0, gt[0]["xywh"], 0.8)]
        result = audit_case(gt, pred, 0)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 1, 0))

    def test_wrong_class_is_not_a_match(self):
        gt = [box(0, (0.5, 0.5, 0.2, 0.2))]
        pred = [box(1, gt[0]["xywh"], 0.9)]
        self.assertEqual(audit_case(gt, pred, 0)["fn"], 1)
        self.assertEqual(audit_case(gt, pred, 1)["fp"], 1)

    def test_low_confidence_candidate_is_a_hint_not_a_match(self):
        gt = [box(0, (0.5, 0.5, 0.2, 0.2))]
        pred = [box(0, gt[0]["xywh"], 0.2)]
        self.assertEqual(audit_case(gt, pred, 0)["fn"], 1)
        self.assertEqual(fn_indicators(gt, pred, 0)[0]["category"], "kandidat di bawah confidence")

    def test_degenerate_saved_prediction_has_zero_iou(self):
        self.assertEqual(box_iou((1, 0.5, 0, 0.5), (0.5, 0.5, 0.2, 0.2)), 0)

    def test_missing_file_is_empty_but_invalid_row_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "labels.txt"
            self.assertEqual(read_boxes(path), [])
            path.write_text("0 0.5 0.5 0 0.2\n")
            with self.assertRaises(ValueError):
                read_boxes(path)


if __name__ == "__main__":
    unittest.main()
