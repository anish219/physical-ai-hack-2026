"""The Meta Quest controllers, through oculus_reader (the open-source Quest app and its adb reader).

Buttons, as the collector uses them:

| Control | Does |
| --- | --- |
| Grip (middle finger), held | Engages the selected arm's clutch (right controller in single-arm mode) |
| Index trigger | That arm's gripper: released opens, pulled closes |
| A | Starts recording an episode; pressed again, saves it |
| B | Discards the episode being recorded |
| Right joystick click | Sends selected arms back to where they started |

@see Quest, QuestState
"""

import dataclasses
import re
import subprocess
import time
from typing import Any

import numpy as np
import numpy.typing as npt

## A controller pose older than this is treated as lost, so a stalled stream releases the clutch.
STALE_AFTER_S = 0.25
## oculus_reader's keys: per side, its pose, grip value and trigger value.
_POSE_KEY = {"right": "r", "left": "l"}
_GRIP_KEY = {"right": "rightGrip", "left": "leftGrip"}
_TRIGGER_KEY = {"right": "rightTrig", "left": "leftTrig"}
## The buttons the collector reacts to.
BUTTONS = ("A", "B", "RJ")
## The teleop app oculus_reader installs on the headset, and the port adb uses over Wi-Fi.
APP = "com.rail.oculus.teleop"
ADB_PORT = 5555
## What `adb devices` says of a headset that is plugged in but adb cannot use, and what to do about it.
ADB_PROBLEMS = {
    "no permissions": (
        "this account cannot open the headset's USB device (over SSH, Ubuntu gives it only to whoever is logged in at "
        "the station's screen): add the Quest's udev rule from docs/hardware_setup.md, step 1, then run "
        "`adb kill-server` and replug the headset"
    ),
    "unauthorized": 'put the headset on and accept "Allow USB debugging" (tick "Always allow from this computer")',
}


class QuestError(RuntimeError):
    """The headset is plugged in but cannot be used yet; the message says why and what to do."""


@dataclasses.dataclass(frozen=True)
class QuestState:
    """One reading of both controllers.

    Attributes:
    - poses: Per side, the controller's `[4, 4]` pose in the Quest's frame, or None when it is not tracked.
    - grip: Per side, the grip button, 0 to 1.
    - trigger: Per side, the index trigger, 0 to 1.
    - pressed: The buttons (BUTTONS) pressed since the last reading.
    """

    poses: dict[str, npt.NDArray[np.float64] | None]
    grip: dict[str, float]
    trigger: dict[str, float]
    pressed: frozenset[str]


class Quest:
    """Reads both controllers from a Quest over adb (USB, or Wi-Fi with `ip_address`)."""

    def __init__(self, ip_address: str | None = None) -> None:
        """Connects to the headset and starts its app; the app is installed on the first run.

        @param ip_address The headset's address for adb over Wi-Fi, or None for USB.
        """
        from oculus_reader.reader import OculusReader

        if ip_address is None:
            problem = adb_problem(adb_devices())
            if problem is not None:
                raise QuestError(f"the Quest is plugged in but adb cannot use it: {problem}")
        self._ip_address = ip_address
        self._reader = OculusReader(ip_address=ip_address)
        self._last_transforms: Any = None
        self._last_change_s = 0.0
        self._ever_changed = False
        self._fresh = False
        self._held: set[str] = set()

    def read(self) -> QuestState:
        """The latest controller state; never blocks."""
        transforms, buttons = self._reader.get_transformations_and_buttons()
        now = time.monotonic()
        if transforms is not self._last_transforms:
            self._last_transforms = transforms
            self._last_change_s = now
            self._ever_changed = self._ever_changed or bool(transforms)
        is_fresh = bool(transforms) and now - self._last_change_s < STALE_AFTER_S
        if is_fresh and not self._fresh:
            # The data just started, or came back after a gap (a restarted app, a headset that slept): a button that
            # is down already was pressed while nothing was read, so it counts as held, never as a new press.
            self._held.clear()
            self._held.update(name for name in BUTTONS if (buttons or {}).get(name) is True)
        self._fresh = is_fresh
        return parse(transforms if is_fresh else {}, buttons or {}, self._held)

    def data_age_s(self) -> float:
        """Seconds since read() last saw new controller data; infinite before the first. Safe from another thread."""
        return time.monotonic() - self._last_change_s if self._ever_changed else float("inf")

    def reconnect(self) -> None:
        """Restarts the teleop app on the headset, in front, and reads it on a new adb log stream.

        For a headset whose data stopped: the app went behind another one (Quest Home), or the log stream was cut
        when the headset slept or its cable came out. It takes a few seconds; read() goes on meanwhile, without data.

        @throws SystemExit, Exception From oculus_reader, when it cannot reach the headset.
        """
        from oculus_reader.reader import OculusReader

        old = self._reader
        old.running = False  # its log thread ends at its next line, at once when its stream is gone
        adb_shell(f"am force-stop {APP}", self._ip_address)
        self._reader = OculusReader(ip_address=self._ip_address)  # starts the app in front, and a new stream
        self._last_transforms, self._last_change_s, self._ever_changed, self._fresh = None, 0.0, False, False
        old.stop()

    def close(self) -> None:
        self._reader.stop()


def adb_problem(adb_devices_output: str) -> str | None:
    """Why adb cannot use the headset on USB, from what `adb devices` prints, or None.

    None when a USB device is ready, or when none is listed at all (oculus_reader then says it found none). Devices
    on Wi-Fi (listed as an address and port) are not looked at.
    """
    states = [state for _, state in usb_devices(adb_devices_output)]
    if "device" in states:
        return None
    for state in states:
        for start, advice in ADB_PROBLEMS.items():
            if state.startswith(start):
                return advice
    return None


def usb_devices(adb_devices_output: str) -> list[tuple[str, str]]:
    """The `(serial, state)` of each USB device `adb devices` lists; one on Wi-Fi (address:port) is left out."""
    found = []
    for line in adb_devices_output.splitlines():
        serial, tab, state = line.partition("\t")
        if tab and serial.count(".") < 3:
            found.append((serial, state.strip()))
    return found


def adb_shell(command: str, ip_address: str | None = None, timeout_s: float = 10.0) -> str:
    """Runs a shell command on the headset over adb, returning its output; nothing when adb does not answer."""
    target = ["-s", f"{ip_address}:{ADB_PORT}"] if ip_address else []
    try:
        result = subprocess.run(["adb", *target, "shell", command], capture_output=True, text=True, timeout=timeout_s)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout


def headset_state(ip_address: str | None = None) -> tuple[bool | None, str | None]:
    """Whether the headset is awake, and the app in front (a package name); None for what adb cannot tell."""
    return parse_headset_state(
        adb_shell(
            "dumpsys power | grep -m1 'mWakefulness='; dumpsys activity activities | grep -m1 'mFocusedApp=Activity'",
            ip_address,
        )
    )


def parse_headset_state(output: str) -> tuple[bool | None, str | None]:
    """headset_state() from its adb output: `mWakefulness=Awake`, `mFocusedApp=ActivityRecord{... <package>/...}`."""
    wake = re.search(r"mWakefulness=(\w+)", output)
    front = re.search(r"mFocusedApp=ActivityRecord\{\S+ \S+ ([\w.]+)/", output)
    return (wake.group(1) == "Awake" if wake else None), (front.group(1) if front else None)


def adb_devices() -> str:
    """What `adb devices` prints, or nothing when adb is not installed or does not answer."""
    try:
        return subprocess.run(["adb", "devices"], capture_output=True, text=True, timeout=15).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def parse(transforms: dict[str, Any], buttons: dict[str, Any], held: set[str]) -> QuestState:
    """Turns one oculus_reader sample into a QuestState.

    @param transforms oculus_reader's poses, `{"r": [4, 4], "l": [4, 4]}`; empty when lost.
    @param buttons oculus_reader's buttons: booleans by name and `(value,)` tuples for the analog controls.
    @param held The buttons held at the last reading; updated in place, so `pressed` holds only new presses.
    @return The state.
    """
    poses = {side: _pose(transforms.get(key)) for side, key in _POSE_KEY.items()}
    down = {name for name in BUTTONS if buttons.get(name) is True}
    pressed = frozenset(down - held)
    held.clear()
    held.update(down)
    return QuestState(
        poses=poses,
        grip={side: _analog(buttons.get(key)) for side, key in _GRIP_KEY.items()},
        trigger={side: _analog(buttons.get(key)) for side, key in _TRIGGER_KEY.items()},
        pressed=pressed,
    )


def _pose(value: Any) -> npt.NDArray[np.float64] | None:
    if value is None:
        return None
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        return None
    return pose


def _analog(value: Any) -> float:
    """An analog control's value, 0 to 1; 0 when the controller does not report it."""
    if not value:
        return 0.0
    return float(min(1.0, max(0.0, value[0])))
