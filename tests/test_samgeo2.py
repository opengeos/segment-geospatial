"""Tests for SamGeo2's per-model autocast handling."""

from unittest import mock

import pytest

torch = pytest.importorskip("torch")

from samgeo import samgeo2  # noqa: E402


def _bare_model(dtype):
    """Build a SamGeo2 without loading weights.

    Args:
        dtype: The autocast dtype to store on the model, or None.

    Returns:
        SamGeo2: A model with only the autocast attribute set.
    """
    model = samgeo2.SamGeo2.__new__(samgeo2.SamGeo2)
    model._autocast_dtype = dtype
    return model


@pytest.fixture
def fake_autocast(monkeypatch):
    """Replace torch.autocast in samgeo2 with a recorder that works without a GPU.

    Returns:
        list: The stack of active autocast dtypes, innermost last.
    """
    active = []

    class _FakeAutocast:
        def __init__(self, device_type, dtype):
            assert device_type == "cuda"
            self.dtype = dtype

        def __enter__(self):
            active.append(self.dtype)

        def __exit__(self, *exc):
            active.pop()

    monkeypatch.setattr(samgeo2.torch, "autocast", _FakeAutocast)
    return active


def test_model_autocast_is_scoped_to_the_call(fake_autocast):
    """Autocast is on inside a wrapped method and off again afterwards."""
    model = _bare_model(torch.float16)

    @samgeo2._with_model_autocast
    def forward(self):
        return list(fake_autocast)

    assert forward(model) == [torch.float16]
    assert fake_autocast == []


def test_two_models_keep_their_own_dtype(fake_autocast):
    """Each model uses its own dtype, whichever was created last."""
    ampere, turing = _bare_model(torch.bfloat16), _bare_model(torch.float16)

    @samgeo2._with_model_autocast
    def dtype_of(self):
        return fake_autocast[-1]

    assert dtype_of(ampere) == torch.bfloat16
    assert dtype_of(turing) == torch.float16
    assert dtype_of(ampere) == torch.bfloat16


def test_no_autocast_without_a_dtype(fake_autocast):
    """An explicit device (no stored dtype) runs without autocast, as before."""
    model = _bare_model(None)

    @samgeo2._with_model_autocast
    def active(self):
        return list(fake_autocast)

    assert active(model) == []


def test_iter_with_model_autocast_only_covers_each_step(fake_autocast):
    """Video propagation computes under autocast but yields outside it."""
    model = _bare_model(torch.float16)

    def frames():
        for i in range(3):
            yield i, list(fake_autocast)

    out = [
        (i, inside, list(fake_autocast))
        for i, inside in model._iter_with_model_autocast(frames())
    ]
    assert out == [(i, [torch.float16], []) for i in range(3)]


def test_init_picks_dtype_only_when_choosing_the_device(monkeypatch):
    """SamGeo2 asks choose_device not to enter a session-wide autocast."""
    choose = mock.Mock(return_value=torch.device("cuda"))
    monkeypatch.setattr(samgeo2.common, "choose_device", choose)
    monkeypatch.setattr(
        samgeo2.common, "get_autocast_dtype", mock.Mock(return_value=torch.float16)
    )
    predictor = mock.Mock()
    sam2_pkg = {
        "sam2": mock.Mock(),
        "sam2.automatic_mask_generator": mock.Mock(),
        "sam2.sam2_image_predictor": mock.Mock(SAM2ImagePredictor=predictor),
        "sam2.sam2_video_predictor": mock.Mock(),
    }
    with mock.patch.dict("sys.modules", sam2_pkg):
        model = samgeo2.SamGeo2(automatic=False)
        explicit = samgeo2.SamGeo2(automatic=False, device="cuda")

    assert choose.call_args.kwargs["enable_autocast"] is False
    assert model._autocast_dtype == torch.float16
    assert explicit._autocast_dtype is None
