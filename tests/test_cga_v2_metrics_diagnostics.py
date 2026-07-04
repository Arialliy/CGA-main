import numpy as np
from metrics import IRSTDMetrics, component_count, target_detection_audit


def test_metrics_basic_positive_case():
    gt = np.zeros((32, 32), dtype=float)
    pred = np.zeros((32, 32), dtype=float)
    gt[10:12, 10:12] = 1
    pred[10:12, 10:12] = 1
    m = IRSTDMetrics(threshold=0.5)
    m.update(pred, gt)
    out = m.get()
    assert out["mIoU"] > 0.99
    assert out["Pd"] > 0.99
    assert component_count(pred) == 1


def test_target_detection_audit_miss():
    gt = np.zeros((32, 32), dtype=float)
    pred = np.zeros((32, 32), dtype=float)
    gt[10:12, 10:12] = 1
    audit = target_detection_audit(pred, gt)
    assert audit["missed_target_count"] == 1
