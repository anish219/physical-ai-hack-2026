"""Right Script panel adapter: launch the headless coordinator and honor panel Stop."""

import os
import pathlib
import signal
import subprocess
import threading

PROJECT = pathlib.Path('/home/pah/physical-ai-hacks')
RUN_ROOT = pathlib.Path('/home/pah/laptop-handoff-20261108/continuous-insertion-review')


class RunHandle:
    def __init__(self, process, cancel):
        self.process = process
        self.finished = threading.Thread(target=self.wait, args=(cancel,), daemon=True)
        self.finished.start()

    def wait(self, cancel):
        self.process.wait()
        cancel.set()  # Tell the Script panel this one attempt has finished.

    def stop(self):
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGINT)
            self.process.wait()


def start(context):
    if context['selection'] != 'right':
        raise ValueError('Use this right-arm insertion adapter in the Right Script panel')
    log_path = pathlib.Path('/tmp/usb-insertion-controller.log')
    with log_path.open('ab') as log:
        process = subprocess.Popen(
            [str(PROJECT / '.venv/bin/python'), '-u', 'scripts/run_usb_insertion.py',
             '--station-config', str(PROJECT / 'configs/station.yaml'), '--script-input'],
            cwd=RUN_ROOT, env={**os.environ, 'PYTHONPATH': str(RUN_ROOT / 'src')},
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    return RunHandle(process, context['cancel'])
