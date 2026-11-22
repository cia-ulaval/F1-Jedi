"""Tests hors ligne du pipeline 3 gestes (aucun capteur requis).

    python -m pytest collect/code_etienne/tests -q
"""

import sys
import types
from pathlib import Path

import numpy as np
import pytest

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

import realtime_test_etienne as rt  # noqa: E402
import trainning_etienne as tr  # noqa: E402
from config import Config  # noqa: E402
from emg_core import OnlineDecoder, ThreeGestureModel, extract_features, frame_signal  # noqa: E402

config = Config()


@pytest.fixture(scope="module")
def recordings():
    return {session: tr.load_recordings([session]) for session in (1, 2)}


@pytest.fixture(scope="module")
def model_s2(recordings):
    return tr.fit_model(recordings[2])


def accuracy(confusion):
    return np.trace(confusion) / confusion.sum()


def test_shape_features_ignore_amplitude():
    rng = np.random.default_rng(0)
    windows = rng.standard_normal((5, config.WINDOW_SIZE))
    rms_1, shape_1 = extract_features(windows, config.N_LAGS)
    rms_2, shape_2 = extract_features(windows * 7.5, config.N_LAGS)
    assert np.allclose(shape_1, shape_2)
    assert np.allclose(rms_2 - rms_1, np.log(7.5))


def test_old_five_gesture_sessions_are_filtered_by_name(recordings):
    # S1 et S2 contiennent 5 gestes ; on ne doit garder que les 3 voulus.
    assert len(recordings[1]) == 3 * config.NUM_REPS
    assert sorted({rec["label"] for rec in recordings[1]}) == [0, 1, 2]


def test_within_session_replay(recordings):
    for session in (1, 2):
        train = [rec for rec in recordings[session] if rec["rep"] in config.TRAIN_REPS]
        test = [rec for rec in recordings[session] if rec["rep"] in config.TEST_REPS]
        assert accuracy(tr.replay_accuracy(tr.fit_model(train), test)) > 0.95


def test_cross_session_replay(recordings):
    # Proxy de la derive : entraine un jour, teste un autre.
    assert accuracy(tr.replay_accuracy(tr.fit_model(recordings[1]), recordings[2])) > 0.90
    assert accuracy(tr.replay_accuracy(tr.fit_model(recordings[2]), recordings[1])) > 0.95


def test_gain_drift_does_not_break_decoding(recordings, model_s2):
    # Gain electrode-peau divise par 2 puis multiplie par 2 : le suivi du
    # repos et les features de forme doivent absorber le changement.
    test = [rec for rec in recordings[2] if rec["rep"] in config.TEST_REPS]
    train = [rec for rec in recordings[2] if rec["rep"] in config.TRAIN_REPS]
    model = tr.fit_model(train)
    for gain in (0.5, 2.0):
        scaled = [dict(rec, signal=rec["signal"] * gain) for rec in test]
        decoder_input = sorted(scaled, key=lambda rec: (rec["rep"], rec["label"]))
        decoder = OnlineDecoder(model, vote_window=config.VOTE_WINDOW, adapt=True)
        rest = [rec for rec in decoder_input if rec["label"] == 0][0]
        decoder.calibrate_rest(frame_signal(rest["signal"][:4000], config.WINDOW_SIZE, config.WINDOW_INC))
        hits = total = 0
        for rec in decoder_input:
            for index, window in enumerate(frame_signal(rec["signal"], config.WINDOW_SIZE, config.WINDOW_INC)):
                label = decoder.update(window)["label"]
                if index >= 20:
                    hits += label == rec["label"]
                    total += 1
        assert hits / total > 0.93, f"gain x{gain}: {hits / total:.3f}"


def test_model_roundtrip(tmp_path, model_s2):
    path = model_s2.save(tmp_path / "model.json")
    loaded = ThreeGestureModel.load(path)
    assert loaded.gestures == config.GESTURES
    assert np.allclose(loaded.weights, model_s2.weights)
    assert loaded.rest_level == model_s2.rest_level


def test_recalibration_recovers_other_session(recordings):
    model = tr.fit_model(recordings[1])
    decoder = OnlineDecoder(model, vote_window=config.VOTE_WINDOW, adapt=True)
    first_rep = {rec["label"]: frame_signal(rec["signal"][:6000], config.WINDOW_SIZE, config.WINDOW_INC)
                 for rec in recordings[2] if rec["rep"] == 0}
    decoder.recalibrate(first_rep[0], first_rep[1], first_rep[2])
    hits = total = 0
    for rec in sorted([r for r in recordings[2] if r["rep"] > 0], key=lambda r: (r["rep"], r["label"])):
        for index, window in enumerate(frame_signal(rec["signal"], config.WINDOW_SIZE, config.WINDOW_INC)):
            label = decoder.update(window)["label"]
            if index >= 20:
                hits += label == rec["label"]
                total += 1
    assert hits / total > 0.95


def test_recalibration_rejects_missing_contraction(recordings, model_s2):
    decoder = OnlineDecoder(model_s2)
    rest = frame_signal([r for r in recordings[2] if r["label"] == 0][0]["signal"], config.WINDOW_SIZE, config.WINDOW_INC)
    with pytest.raises(ValueError):
        decoder.recalibrate(rest, rest, rest)


class FakeODH:
    """Imite OnlineDataHandler : tampon range du plus recent au plus ancien."""

    def __init__(self, signal, buffer_size=2000):
        self.signal, self.position, self.buffer_size = signal, 0, buffer_size
        self.offset = 0

    def advance(self, samples):
        self.position = min(self.position + samples, len(self.signal))

    def reset(self):
        self.offset = self.position

    def get_data(self, N=0):
        fresh = self.signal[self.offset:self.position][-self.buffer_size:]
        buffer = np.zeros((self.buffer_size, 1))
        buffer[:len(fresh), 0] = fresh[::-1]
        count = np.array([[self.position - self.offset]], dtype=np.int32)
        return {"emg": buffer[:N] if N else buffer}, {"emg": count}


def test_window_reader_returns_chronological_latest_window():
    signal = np.arange(5000, dtype=float)
    odh = FakeODH(signal)
    reader = rt.WindowReader(odh, 400, 40)
    odh.advance(399)
    assert reader.poll() is None           # pas encore une fenetre complete
    odh.advance(1)
    assert np.array_equal(reader.poll(), signal[:400])
    odh.advance(39)
    assert reader.poll() is None           # increment incomplet
    odh.advance(1)
    assert np.array_equal(reader.poll(), signal[40:440])
    odh.advance(90)                        # paquet de 90 : deux fenetres, pas de 40 exact
    assert np.array_equal(reader.poll(), signal[80:480])
    assert np.array_equal(reader.poll(), signal[120:520])
    assert reader.poll() is None
    odh.advance(500)                       # gros retard : on saute a la plus recente
    assert np.array_equal(reader.poll(), signal[630:1030])
    reader.reset()
    odh.advance(100)
    assert reader.poll() is None           # apres reset, on attend 400 echantillons frais


def test_game_bridge_with_fake_gamepad(monkeypatch, tmp_path, recordings):
    import realtime_game_bridge as bridge

    events = []

    class FakePad:
        def __init__(self):
            self.state = {"a": False, "b": False, "x": 0}

        def reset(self):
            self.state = {"a": False, "b": False, "x": 0}

        def press_button(self, button):
            self.state[button] = True

        def left_joystick(self, x_value, y_value):
            self.state["x"] = x_value

        def update(self):
            events.append(dict(self.state))

    fake_vg = types.SimpleNamespace(
        VX360Gamepad=FakePad,
        XUSB_BUTTON=types.SimpleNamespace(XUSB_GAMEPAD_A="a", XUSB_GAMEPAD_B="b", XUSB_GAMEPAD_X="xbtn"),
    )
    pressed = {"o"}
    fake_keyboard = types.SimpleNamespace(is_pressed=lambda key: key in pressed)
    monkeypatch.setattr(bridge, "vg", fake_vg)
    monkeypatch.setattr(bridge, "keyboard", fake_keyboard)
    monkeypatch.setattr(rt.config, "REST_CALIBRATION_S", 0.0)
    monkeypatch.setattr(rt.ReplayReader, "collect", lambda self, duration_s, on_tick=None: np.empty((0, self.window_size)))

    model_path = tr.fit_model(recordings[2]).save(tmp_path / "f1jedi_3g_test.json")
    model = ThreeGestureModel.load(model_path)
    reader = rt.ReplayReader(2, model.gestures, model.window_size, model.window_inc, model.fs, speed=40.0)
    bridge.run_realtime_game(str(model_path), reader=reader)

    steer = np.array([event["x"] for event in events])
    assert events[-1] == {"a": False, "b": False, "x": 0}      # manette relachee a la fin
    assert any(event["a"] for event in events)                 # accelerateur maintenu en conduite
    assert (steer < 0).any() and (steer > 0).any() and (steer == 0).any()


def test_rest_calibration_is_capped_when_user_is_contracting(recordings, model_s2):
    decoder = OnlineDecoder(model_s2)
    strong = frame_signal([r for r in recordings[2] if r["label"] == 2][0]["signal"], config.WINDOW_SIZE, config.WINDOW_INC)
    decoder.calibrate_rest(strong)
    assert decoder.rest_level <= model_s2.rest_level + np.log(3.0) + 1e-9


class FakeStreamer:
    """Imite le processus SiFiBridgeStreamer de libemg."""

    def __init__(self, stops_on_signal):
        import threading
        self.signal = threading.Event()
        self.pid = 4242
        self._alive = True
        self._stops_on_signal = stops_on_signal

    def is_alive(self):
        return self._alive

    def join(self, timeout=None):
        if self._stops_on_signal and self.signal.is_set():
            self._alive = False

    def terminate(self):
        self._alive = False


def test_prepare_streamer_keeps_handle_and_stop_is_clean(monkeypatch):
    # Bug : prepare_streamer jetait le processus du streamer, impossible de
    # l'arreter proprement -> sifibridge pouvait rester accroche au Bluetooth.
    import collect_etienne as ce

    streamer = FakeStreamer(stops_on_signal=True)
    fake_libemg = types.SimpleNamespace(
        streamers=types.SimpleNamespace(sifi_biopoint_streamer=lambda **kwargs: (streamer, kwargs["shared_memory_items"])),
        data_handler=types.SimpleNamespace(OnlineDataHandler=lambda shared_memory_items: "odh"),
    )
    monkeypatch.setitem(sys.modules, "libemg", fake_libemg)
    monkeypatch.setattr(ce, "_ACTIVE_STREAMERS", [])
    calls = []
    monkeypatch.setattr(ce.subprocess, "run", lambda *a, **k: calls.append(a))

    assert ce.prepare_streamer() == "odh"                      # ancienne interface inchangee
    assert ce._ACTIVE_STREAMERS == [streamer]                  # arrete a la sortie (atexit)
    got_streamer, odh = ce.prepare_streamer(return_streamer=True)
    assert got_streamer is streamer and odh == "odh"
    ce.stop_streamer(streamer)
    assert streamer.signal.is_set() and not streamer.is_alive()
    assert calls == []                                         # arret normal, pas de kill


def test_stop_streamer_kills_tree_when_sensor_is_silent(monkeypatch):
    # Capteur eteint : libemg n'execute son nettoyage qu'au paquet suivant,
    # qui n'arrive jamais. On doit tuer streamer + sifibridge.
    import collect_etienne as ce

    streamer = FakeStreamer(stops_on_signal=False)
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        streamer._alive = False

    monkeypatch.setattr(ce.subprocess, "run", fake_run)
    monkeypatch.setattr(ce.sys, "platform", "win32")
    monkeypatch.setattr(ce, "_ACTIVE_STREAMERS", [streamer])
    ce.stop_streamer(streamer, timeout_s=0.01)
    assert calls == [["taskkill", "/F", "/T", "/PID", "4242"]]
    assert ce._ACTIVE_STREAMERS == []


def test_replay_reader_yields_every_window_even_when_accelerated(monkeypatch):
    # Bug : a vitesse x20, ReplayReader sautait ~4 fenetres sur 5 et notait la
    # sortie contre la verite du moment present (85.9 % au lieu de 94.0 % en x1).
    clock = {"t": 0.0}
    monkeypatch.setattr(rt.time, "perf_counter", lambda: clock["t"])
    reader = rt.ReplayReader(2, config.GESTURES, config.WINDOW_SIZE, config.WINDOW_INC, config.EMG_FS, speed=20.0)
    clock["t"] = 1.0                                       # 40 000 echantillons d'un coup
    ends = []
    while (window := reader.poll()) is not None:
        ends.append(reader._last_end)
        assert np.array_equal(window, reader.signal[reader._last_end - config.WINDOW_SIZE:reader._last_end])
        assert reader.current_truth() == reader.truth[reader._last_end - 1]
    assert np.all(np.diff(ends) == config.WINDOW_INC)
    assert ends[-1] == 40000
