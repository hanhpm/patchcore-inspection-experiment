import pytest

from patchcore.modules_pafa.diagnostics import compute_e3_diagnostics


def test_compute_e3_diagnostics_records_required_metrics():
    diagnostics = compute_e3_diagnostics(
        scores=[0.1, 0.2, 0.8, 0.9],
        anomaly_labels=[False, False, True, True],
        adapter_displacement=0.03,
        aupro_005=0.44,
    )

    assert diagnostics["adapter_displacement"] == pytest.approx(0.03)
    assert diagnostics["nominal_stability"]["score_p95"] == pytest.approx(0.195)
    assert diagnostics["nominal_stability"]["score_p99"] == pytest.approx(0.199)
    assert diagnostics["real_gt_margin"] == pytest.approx(0.7)
    assert diagnostics["au_pro_0.05"] == pytest.approx(0.44)
