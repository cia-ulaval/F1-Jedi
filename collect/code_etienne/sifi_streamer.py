"""Streamer BioPoint : celui de libemg, avec un arret qui ne met pas le capteur en veille.

L'arret de libemg 2.0.3 (SiFiBridgeStreamer.cleanup) :
  - envoie 'deep-sleep' au BioPoint : mesure sur capteur, il ne se reconnecte
    plus ensuite (6 essais "Could not connect... Retrying" en 30 s) tant qu'on
    n'appuie pas sur son bouton ;
  - plante sur disconnect()["connected"] (sifi_bridge_py 1.3 retourne un bool)
    avant d'avoir tue sifibridge.

Module separe pour que la classe soit importable par le processus enfant
(multiprocessing en mode spawn sous Windows).
"""

from multiprocessing import Lock

from libemg._streamers._sifi_bridge_streamer import SiFiBridgeStreamer


class SiFiStreamerNoSleep(SiFiBridgeStreamer):
    """Arret : stop acquisition -> deconnexion -> sifibridge tue. Pas de veille."""

    deep_sleep_on_exit = False

    def disconnect(self):
        status = self.sb.disconnect()
        self.connected = status["connected"] if isinstance(status, dict) else bool(status)
        return self.connected

    def cleanup(self):
        steps = [("acquisition arretee", self.stop_sampling), ("deconnecte", self.disconnect)]
        if self.deep_sleep_on_exit:
            steps.insert(1, ("capteur en veille", self.deep_sleep))
        for label, step in steps:
            try:
                step()
                print(f"[streamer] {label}")
            except Exception as error:      # on continue : sifibridge doit etre tue
                print(f"[streamer] {label} : echec ({error})")
        try:
            self.sb._bridge.kill()
            print("[streamer] sifibridge arrete")
        finally:
            self.smm.cleanup()


def start_biopoint_streamer(name, shared_memory_items, notch, streaming=False, deep_sleep_on_exit=False):
    """Equivalent de libemg.streamers.sifi_biopoint_streamer (EMG seul, filtrage capteur)."""
    for item in shared_memory_items:
        item.append(Lock())
    streamer = SiFiStreamerNoSleep(
        name, shared_memory_items,
        ecg=False, emg=True, eda=False, imu=False, ppg=False,
        filtering=True, emg_notch_freq=notch, emg_bandpass=(20, 450),
        eda_bandpass=(0, 5), eda_freq=0, streaming=streaming, mac=None,
    )
    streamer.deep_sleep_on_exit = deep_sleep_on_exit
    streamer.start()
    return streamer, shared_memory_items
