"""Leonard phase/subphase sequences executed by the station's existing API workers.

The original bridge and editor are retained under vendor/leonard_sequences.
Dream and wait windows advance by time, exactly as in that editor; completion
does not assert that a manipulation succeeded. Each subphase can attach a
monitor (VLM prompt or registered script pointer). Monitors are evaluated after
the window; they do not yet gate advancement. See docs/sequence_monitors.md.
"""

import copy
import json
import math
import pathlib
import threading
import time
import uuid

MONITOR_TYPES = ("none", "prompt", "script")


def identifier():
    return uuid.uuid4().hex


def default_monitor():
    return {"type": "none", "prompt": "", "script": ""}


def default_definitions():
    return {
        "left": {"phases": []},
        "right": {
            "phases": [
                {
                    "id": identifier(),
                    "title": "Unplug USB",
                    "steps": [
                        {
                            "id": identifier(),
                            "type": "dream",
                            "label": "Unplug the USB",
                            "prompt": "Unplug the USB from the port, then move it clear of the workspace.",
                            "modelId": "dreamlabs/dream-1-beta",
                            "durationSeconds": 20,
                            "monitor": default_monitor(),
                        }
                    ],
                }
            ]
        },
        "both": {"phases": []},
    }


def validate_monitor(step, *, for_run=False, scripts=()):
    monitor = step.get("monitor")
    if monitor is None:
        step["monitor"] = default_monitor()
        return
    if not isinstance(monitor, dict) or monitor.get("type") not in MONITOR_TYPES:
        raise ValueError("Monitor type must be none, prompt, or script")
    prompt = monitor.get("prompt", "")
    script = monitor.get("script", "")
    if not isinstance(prompt, str) or len(prompt) > 1000:
        raise ValueError("Monitor prompts must be text up to 1000 characters")
    if not isinstance(script, str) or len(script) > 160:
        raise ValueError("Monitor script pointers must be text up to 160 characters")
    if for_run and monitor["type"] == "prompt" and not prompt.strip():
        raise ValueError("Enter a VLM monitor prompt, or set monitor type to none")
    if monitor["type"] == "script":
        pointer = script.strip()
        if for_run and not pointer:
            raise ValueError("Select a registered monitor script pointer")
        if for_run and pointer not in scripts:
            raise ValueError(f"Unknown monitor script pointer: {pointer}")
    step["monitor"] = {"type": monitor["type"], "prompt": prompt, "script": script.strip()}


def validate_definition(value, *, for_run=False, scripts=()):
    if not isinstance(value, dict) or not isinstance(value.get("phases"), list):
        raise ValueError("A sequence needs an ordered phases list")
    definition = copy.deepcopy(value)
    for phase in definition["phases"]:
        if not isinstance(phase, dict) or not isinstance(phase.get("steps"), list):
            raise ValueError("Each phase needs a subphase list")
        if not isinstance(phase.get("id"), str) or not isinstance(phase.get("title"), str):
            raise ValueError("Each phase needs an ID and title")
        for step in phase["steps"]:
            if not isinstance(step, dict) or step.get("type") not in ("dream", "wait"):
                raise ValueError("Subphases must be Dream tasks or wait checkpoints")
            if not isinstance(step.get("id"), str) or not isinstance(step.get("label"), str):
                raise ValueError("Each subphase needs an ID and name")
            duration = step.get("durationSeconds")
            minimum = 1 if step["type"] == "dream" else 0
            if type(duration) not in (int, float) or not math.isfinite(duration) or not minimum <= duration <= 300:
                raise ValueError(f"{step['type'].title()} window must be {minimum}–300 seconds")
            if step["type"] == "dream":
                if not isinstance(step.get("prompt"), str) or len(step["prompt"]) > 1000:
                    raise ValueError("Dream prompts must be text up to 1000 characters")
                if (
                    not isinstance(step.get("modelId"), str)
                    or not step["modelId"].strip()
                    or len(step["modelId"]) > 160
                ):
                    raise ValueError("Enter a Dream checkpoint ID up to 160 characters")
                if for_run and not step["prompt"].strip():
                    raise ValueError("Enter a prompt for every Dream subphase")
            validate_monitor(step, for_run=for_run, scripts=scripts)
            step.pop("state", None)
    if for_run and not any(phase["steps"] for phase in definition["phases"]):
        raise ValueError("Add at least one subphase before running the sequence")
    return definition


class MonitorRegistry:
    """Plug-in point for Anish's VLM prompt monitor and named script monitors."""

    def __init__(self):
        self.scripts = {}
        self.prompt_evaluator = None

    def register_script(self, pointer, evaluate, *, label=None):
        if not isinstance(pointer, str) or not pointer.strip() or len(pointer) > 160:
            raise ValueError("Script pointers must be 1–160 character names")
        if not callable(evaluate):
            raise ValueError("Monitor scripts must be callable")
        self.scripts[pointer.strip()] = {"evaluate": evaluate, "label": label or pointer.strip()}

    def set_prompt_evaluator(self, evaluate):
        if evaluate is not None and not callable(evaluate):
            raise ValueError("Prompt evaluator must be callable")
        self.prompt_evaluator = evaluate

    def pointers(self):
        return tuple(self.scripts)

    def available_scripts(self):
        return [{"pointer": pointer, "label": item["label"]} for pointer, item in self.scripts.items()]

    def evaluate(self, monitor, context):
        kind = (monitor or {}).get("type") or "none"
        if kind == "none":
            return {"result": "skipped"}
        if kind == "prompt":
            if self.prompt_evaluator is None:
                return {"result": "unwired", "detail": "No VLM prompt evaluator registered"}
            return self._normalize(self.prompt_evaluator(monitor.get("prompt", ""), context))
        if kind == "script":
            pointer = (monitor.get("script") or "").strip()
            item = self.scripts.get(pointer)
            if item is None:
                return {"result": "unwired", "detail": f"No script registered for {pointer!r}"}
            return self._normalize(item["evaluate"](context))
        return {"result": "unwired", "detail": f"Unknown monitor type {kind!r}"}

    def _normalize(self, result):
        if isinstance(result, str):
            result = {"result": result}
        if not isinstance(result, dict) or result.get("result") not in (
            "yes",
            "no",
            "uncertain",
            "skipped",
            "unwired",
        ):
            raise ValueError("Monitors must return yes, no, uncertain, skipped, or unwired")
        return dict(result)


class SequenceRun:
    def __init__(self, selection, caller, nodes, changed, monitors):
        self.selection, self.caller, self.nodes, self.changed = selection, caller, nodes, changed
        self.monitors = monitors
        self.sides = ("left", "right") if selection == "both" else (selection,)
        self.lock, self.gate = threading.RLock(), threading.Lock()
        self.cancel = threading.Event()
        self.thread = None
        self.action_active = False
        self.state = dict(
            state="idle",
            busy=False,
            running=False,
            phase_index=None,
            step_index=None,
            phase_title="",
            step_label="",
            error="",
            step_started=None,
            durationSeconds=0,
            monitor_result=None,
        )

    def snapshot(self):
        with self.lock:
            state = dict(self.state)
        started = state.pop("step_started")
        state["elapsedSeconds"] = min(state["durationSeconds"], max(0, time.monotonic() - started)) if started else 0
        return state

    def report(self, **values):
        with self.lock:
            self.state.update(values)
        self.changed()

    def hold(self):
        for side in self.sides:
            node = self.nodes[side]
            if node.snapshot()["arm_open"]:
                node.control.post("/stop", {})

    def start(self, definition):
        definition = validate_definition(definition, for_run=True, scripts=self.monitors.pointers())
        with self.gate:
            if self.snapshot()["busy"]:
                raise ValueError("This sequence is already running or finishing a call")
            self.cancel = threading.Event()
            self.action_active = False
            self.report(
                state="running",
                busy=True,
                running=True,
                error="",
                phase_index=None,
                step_index=None,
                phase_title="",
                step_label="",
                step_started=None,
                durationSeconds=0,
                monitor_result=None,
            )
            self.thread = threading.Thread(
                target=self.run, args=(definition, self.cancel), name=f"sequence:{self.selection}", daemon=True
            )
            self.thread.start()

    def stop(self):
        with self.gate:
            if not self.snapshot()["busy"]:
                return
            self.cancel.set()
            if self.action_active:
                self.caller.cancel_playback()
                self.action_active = False
                self.hold()
            self.report(state="stopped", running=False)

    def dream_step(self, step, cancel):
        self.caller.connect()
        while not cancel.is_set():
            state = self.caller.snapshot()
            if state.get("observation_error"):
                raise RuntimeError(state["observation_error"])
            if state.get("observation") is not None:
                break
            cancel.wait(0.05)
        with self.gate:
            if cancel.is_set():
                return
            self.caller.start({"task": step["prompt"], "checkpoint": step["modelId"]}, run=True)
            self.action_active = True
        deadline = time.monotonic() + step["durationSeconds"]
        while not cancel.is_set() and time.monotonic() < deadline:
            state = self.caller.snapshot()
            if state.get("error") or not state["busy"]:
                raise RuntimeError(state.get("error") or "Dream step stopped before its window finished")
            cancel.wait(min(0.1, max(0, deadline - time.monotonic())))
        with self.gate:
            if cancel.is_set():
                return
            self.caller.cancel_playback()
            self.hold()
        # The cancelled in-flight reply must unwind before another step opens a policy.
        while self.caller.snapshot()["busy"] and not cancel.is_set():
            cancel.wait(0.05)
        with self.gate:
            self.action_active = False

    def observation_context(self, step):
        arms, camera_urls = {}, {}
        for side, node in self.nodes.items():
            arms[side] = node.snapshot()
            port = getattr(node, "port", None)
            if port:
                camera_urls[side] = f"http://127.0.0.1:{port}"
        return {
            "selection": self.selection,
            "sides": self.sides,
            "step": copy.deepcopy(step),
            "arms": arms,
            "camera_urls": camera_urls,
        }

    def evaluate_monitor(self, step):
        result = self.monitors.evaluate(step.get("monitor"), self.observation_context(step))
        self.report(monitor_result=result)
        return result

    def run(self, definition, cancel):
        try:
            for phase_index, phase in enumerate(definition["phases"]):
                for step_index, step in enumerate(phase["steps"]):
                    if cancel.is_set():
                        return
                    self.report(
                        phase_index=phase_index,
                        step_index=step_index,
                        phase_title=phase["title"],
                        step_label=step["label"],
                        step_started=time.monotonic(),
                        durationSeconds=step["durationSeconds"],
                    )
                    if step["type"] == "wait":
                        cancel.wait(step["durationSeconds"])
                    else:
                        self.dream_step(step, cancel)
                    if cancel.is_set():
                        return
                    self.evaluate_monitor(step)
            self.report(state="complete", running=False)
        except Exception as error:
            with self.gate:
                if not cancel.is_set():
                    if self.action_active:
                        self.caller.cancel_playback()
                        self.action_active = False
                        self.hold()
                    self.report(state="error", running=False, error=str(error))
        finally:
            self.report(busy=False, running=False)


class SequenceManager:
    def __init__(self, callers, nodes, publish, path=None, monitors=None):
        self.lock, self.publish = threading.RLock(), publish
        self.monitors = monitors or MonitorRegistry()
        if monitors is None:
            try:
                from dream_hackathon.monitor_scripts import register_all

                register_all(self.monitors)
            except ImportError:
                pass
        self.path = pathlib.Path(path).expanduser() if path is not None else None
        self.definitions = default_definitions()
        if self.path is not None and self.path.exists():
            saved = json.loads(self.path.read_text(encoding="utf-8"))
            self.definitions.update(
                {side: validate_definition(value, scripts=self.monitors.pointers()) for side, value in saved.items()}
            )
        self.revisions = dict.fromkeys(self.definitions, 0)
        self.runs = {side: SequenceRun(side, callers[side], nodes, self.changed, self.monitors) for side in callers}

    def snapshot(self):
        with self.lock:
            definitions, revisions = copy.deepcopy(self.definitions), dict(self.revisions)
        return {
            side: {
                "definition": definition,
                "revision": revisions[side],
                **(self.runs[side].snapshot() if side in self.runs else {"state": "idle", "busy": False}),
            }
            for side, definition in definitions.items()
        }

    def changed(self):
        self.publish({"type": "sequences", "sequences": self.snapshot()})

    def update(self, selection, definition):
        if selection not in self.definitions:
            raise ValueError("Choose left, right, or both for Sequences")
        definition = validate_definition(definition, scripts=self.monitors.pointers())
        with self.lock:
            self.definitions[selection] = definition
            self.revisions[selection] += 1
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.write_text(json.dumps(self.definitions, indent=2), encoding="utf-8")
        self.changed()

    def start(self, selection):
        with self.lock:
            definition = copy.deepcopy(self.definitions[selection])
        self.runs[selection].start(validate_definition(definition, for_run=True, scripts=self.monitors.pointers()))

    def stop_sides(self, sides):
        for run in self.runs.values():
            if set(sides).intersection(run.sides):
                run.stop()

    def close(self):
        self.stop_sides(("left", "right"))
