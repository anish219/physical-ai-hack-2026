"""Print V2 CV port observations from existing HTTP cameras. GET only, no motion."""

import argparse
import json
import time
import uuid

from dream_hackathon.usb_port_detector import DEFAULT_SCRIPTS, UsbPortDetector, load_cv_detector


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scripts-path", default=DEFAULT_SCRIPTS)
    p.add_argument("--calibration", default=None)
    p.add_argument("--scene-url", default="http://127.0.0.1:8692/overhead.jpg")
    p.add_argument(
        "--wrist-url", default="http://127.0.0.1:8692/wrist.jpg", help="Empty string disables this view explicitly"
    )
    p.add_argument("--run-id", default=None)
    p.add_argument("--iterations", type=int, default=1)
    p.add_argument("--interval", type=float, default=0.2)
    args = p.parse_args()
    if args.iterations < 1 or not 0.1 <= args.interval <= 60:
        p.error("Use positive iterations and an interval between .1 and 60 seconds")
    run_id = args.run_id or "ports-read-only-" + uuid.uuid4().hex
    detector, fetch = load_cv_detector(args.scripts_path)
    if args.calibration:
        import numpy as np

        with np.load(args.calibration, allow_pickle=False) as loaded:
            detector.templates = {key: loaded[key] for key in loaded.files}
    ports = UsbPortDetector(run_id, detector=detector)
    for _ in range(args.iterations):
        started = time.monotonic()
        images, stamps, _ages, errors = fetch(args.scene_url or None, args.wrist_url or None, run_id)
        if errors:
            # Do not silently drop a configured camera when it fails.
            result = ports._unknown("camera_transport_failed", stamps)
        else:
            result = ports.observe(
                rgb_scene=images.get("scene"), rgb_wrist=images.get("wrist"), stamps=stamps, now=time.monotonic()
            )
        print(json.dumps(dict(result, camera_errors=errors)), flush=True)
        if _ + 1 < args.iterations:
            time.sleep(max(0, args.interval - (time.monotonic() - started)))


if __name__ == "__main__":
    main()
