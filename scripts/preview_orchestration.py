"""Local UI preview with simulated arm status. Opens no robot devices or cameras."""

import argparse
import threading

from dream_hackathon.so101_dashboard import Controls
from dream_hackathon.unified_control import UnifiedController, UnifiedPage


class PreviewNode:
    port = 0

    def snapshot(self):
        return {"arm_open": False, "grip_torque_nm": 0.5, "message": "Local UI preview · no arm attached"}

    def cancel_home(self):
        pass


class PreviewLeader:
    side = None

    def state(self):
        return Controls(1, "preview").snapshot()

    def select(self, side):
        self.side = side

    def action(self, path, body):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8096)
    args = parser.parse_args()
    controller = UnifiedController({side: PreviewNode() for side in ("left", "right")}, PreviewLeader())
    page = UnifiedPage(controller, args.port)
    print(f"Local preview: http://127.0.0.1:{page.port}", flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        page.close()


if __name__ == "__main__":
    main()
