"""Editable flow graphs with concurrent operations and explicit event/time triggers."""

import copy
import json
import math
import pathlib
import re
import subprocess
import tempfile
import threading
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor

TYPES = {"start", "trigger", "action", "condition", "loop", "fork", "join", "end"}
PORTS = {
    "start": {"next"},
    "trigger": {"next", "timeout"},
    "action": {"next", "error", "timeout"},
    "condition": {"true", "false"},
    "loop": {"body", "done"},
    "fork": {"next"},
    "join": {"next"},
    "end": set(),
}


def default_graph():
    return {
        "nodes": [
            {"id": "go", "type": "start", "label": "Go", "x": 80, "y": 100, "config": {}},
            {"id": "end", "type": "end", "label": "Finish", "x": 420, "y": 100, "config": {}},
        ],
        "edges": [{"id": "go-end", "source": "go", "port": "next", "target": "end"}],
    }


def number(value, label, minimum=0):
    if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
        raise ValueError(f"{label} must be a finite number >= {minimum}")
    return value


def validate_condition(condition):
    if not isinstance(condition, dict):
        raise ValueError("A trigger needs a condition object")
    kind = condition.get("kind")
    if kind in ("all", "any"):
        children = condition.get("conditions")
        if not isinstance(children, list) or not children:
            raise ValueError("All/any needs at least one nested condition")
        for child in children:
            validate_condition(child)
    elif kind == "not":
        validate_condition(condition.get("condition"))
    elif kind in ("delay", "elapsed"):
        if not isinstance(condition.get("secondsRef"), str):
            number(condition.get("seconds"), "Trigger seconds")
    elif kind in ("signal", "vlm", "result", "variable"):
        if not isinstance(condition.get("name"), str) or not condition["name"]:
            raise ValueError("A data condition needs a signal, node or variable name")
        if condition.get("op", "eq") not in ("eq", "ne", "gt", "gte", "lt", "lte", "contains", "exists"):
            raise ValueError("Choose a supported comparison")
        if not isinstance(condition.get("path", ""), str):
            raise ValueError("Comparison path must be text")
    elif kind == "script_end":
        if not isinstance(condition.get("name"), str) or not condition["name"]:
            raise ValueError("Script completion needs an operation node ID")
        number(condition.get("count", 1), "Script completion count", 1)
    else:
        raise ValueError(f"Unknown trigger kind: {kind}")


def validate_graph(value, selection, *, for_run=False):
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("nodes"), list)
        or not isinstance(value.get("edges"), list)
    ):
        raise ValueError("A flow needs nodes and edges lists")
    graph = copy.deepcopy(value)
    nodes = {}
    for node in graph["nodes"]:
        if not isinstance(node, dict) or not isinstance(node.get("id"), str) or not node["id"] or node["id"] in nodes:
            raise ValueError("Every block needs a unique nonempty ID")
        nodes[node["id"]] = node
        kind = node.get("type")
        if kind not in TYPES or not isinstance(node.get("label"), str):
            raise ValueError("Choose a block type and label")
        number(node.get("x", 0), "Block x")
        number(node.get("y", 0), "Block y")
        config = node.setdefault("config", {})
        if not isinstance(config, dict):
            raise ValueError("Block configuration must be an object")
        if kind in ("trigger", "condition"):
            validate_condition(config.get("condition"))
        if kind == "loop":
            if config.get("mode", "count") == "count":
                count = config.get("count", 1)
                number(count, "Loop count")
                if int(count) != count:
                    raise ValueError("Loop count must be a whole number")
            elif config.get("mode") == "while":
                validate_condition(config.get("condition"))
            else:
                raise ValueError("Loop mode must be count or while")
        if "timeoutSeconds" in config and config["timeoutSeconds"] is not None:
            number(config["timeoutSeconds"], "Configured timeout")
        if kind == "action":
            operation = config.get("operation")
            if operation not in (
                "script",
                "http",
                "dream",
                "vlm",
                "episode",
                "binding",
                "dummy",
                "finish",
                "emit",
                "set",
            ):
                raise ValueError("Choose script, HTTP API, Dream API, VLM, signal or variable operation")
            if operation == "script":
                argv = config.get("argv")
                if not isinstance(argv, list) or not argv or any(not isinstance(arg, str) for arg in argv):
                    raise ValueError("Script command must be a nonempty JSON array of arguments")
            if operation == "http" and not isinstance(config.get("url"), str):
                raise ValueError("HTTP operation needs a URL")
            if operation == "episode" and not config.get("planPath"):
                raise ValueError("Recorded episode needs its plan path")
            if operation == "binding":
                binding = graph.get("bindings", {}).get(config.get("binding"))
                if not isinstance(binding, dict) or binding.get("operation") == "binding":
                    raise ValueError("Operation needs a defined, nonrecursive binding")
                probe = {
                    "nodes": [{**node, "config": {**binding, "selection": config.get("selection", "both")}}],
                    "edges": [],
                }
                validate_graph(probe, selection)
            if "selection" in config and config["selection"] not in ("left", "right", "both"):
                raise ValueError("Choose the operation's target arms")
            if operation == "vlm":
                if config.get("provider") not in ("openai", "ollama") or not config.get("model"):
                    raise ValueError("Choose a VLM provider and model")
                if not config.get("question") or not (config.get("imageUrl") or config.get("imagePath")):
                    raise ValueError("VLM needs a question and image URL or local path")
            if operation == "dream":
                target = config.get("selection", selection)
                if target not in ("left", "right", "both") or (selection != "both" and target != selection):
                    raise ValueError("Dream target must belong to this flow's arm selection")
                if not isinstance(config.get("task"), str) or not config["task"].strip():
                    raise ValueError("Dream operation needs a task")
                if config.get("run", True):
                    number(config.get("durationSeconds"), "Dream playback window", 0.001)
            if operation in ("emit", "set") and not config.get("name"):
                raise ValueError("Signal/variable operation needs a name")
    edge_ids = set()
    for edge in graph["edges"]:
        if not isinstance(edge, dict) or not isinstance(edge.get("id"), str) or edge["id"] in edge_ids:
            raise ValueError("Every connection needs a unique ID")
        edge_ids.add(edge["id"])
        if edge.get("source") not in nodes or edge.get("target") not in nodes:
            raise ValueError("Connection refers to a missing block")
        if edge.get("port") not in PORTS[nodes[edge["source"]]["type"]]:
            raise ValueError("Connection uses an unsupported output port")
        if nodes[edge["target"]]["type"] == "start":
            raise ValueError("Go cannot have an incoming connection")
    if for_run:
        if sum(node["type"] == "start" for node in nodes.values()) != 1:
            raise ValueError("A runnable flow needs exactly one Go block")
        for node in nodes.values():
            for port in PORTS[node["type"]] - {"error", "timeout"}:
                edges = [edge for edge in graph["edges"] if edge["source"] == node["id"] and edge["port"] == port]
                if not edges:
                    raise ValueError(f"Connect {node['label']} / {port}")
                if node["type"] != "fork" and len(edges) > 1:
                    raise ValueError("Use a Parallel block to fan out an output")
    return graph


MISSING = object()


def lookup(value, path):
    for part in path.split(".") if path else []:
        if isinstance(value, dict):
            value = value.get(part, MISSING)
        elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
        else:
            return MISSING
    return value


def compare(actual, op, expected):
    if op == "exists":
        return actual is not MISSING
    if actual is MISSING:
        return False
    try:
        if op == "eq":
            return actual == expected
        if op == "ne":
            return actual != expected
        if op == "gt":
            return actual > expected
        if op == "gte":
            return actual >= expected
        if op == "lt":
            return actual < expected
        if op == "lte":
            return actual <= expected
        if op == "contains":
            return expected in actual
    except TypeError:
        return False
    return False


def evaluate(condition, context, elapsed, active_elapsed):
    kind = condition["kind"]
    if kind == "all":
        return all(evaluate(child, context, elapsed, active_elapsed) for child in condition["conditions"])
    if kind == "any":
        return any(evaluate(child, context, elapsed, active_elapsed) for child in condition["conditions"])
    if kind == "not":
        return not evaluate(condition["condition"], context, elapsed, active_elapsed)
    if kind in ("delay", "elapsed"):
        seconds = lookup(context, condition["secondsRef"]) if "secondsRef" in condition else condition["seconds"]
        return seconds is not MISSING and (active_elapsed if kind == "delay" else elapsed) >= seconds
    if kind == "script_end":
        result = context["results"].get(condition["name"], {})
        return result.get("operation") == "script" and result.get("executions", 0) >= condition.get("count", 1)
    group = {"signal": "signals", "vlm": "signals", "result": "results", "variable": "variables"}[kind]
    actual = lookup(context[group].get(condition["name"], MISSING), condition.get("path", ""))
    expected = lookup(context, condition["valueRef"]) if "valueRef" in condition else condition.get("value", True)
    return expected is not MISSING and compare(actual, condition.get("op", "eq"), expected)


def expand(value, context):
    """{{results.node.verdict}}, {{signals.name}}, {{variables.name}} and {{env.KEY}}."""
    import os

    def resolve(path):
        root, _, tail = path.partition(".")
        result = lookup(dict(os.environ) if root == "env" else context.get(root, {}), tail)
        if result is MISSING:
            raise ValueError(f"No value for template: {path}")
        return result

    if isinstance(value, str):
        match = re.fullmatch(r"\{\{([\w.]+)\}\}", value)
        if match:
            return resolve(match[1])
        return re.sub(r"\{\{([\w.]+)\}\}", lambda match: str(resolve(match[1])), value)
    if isinstance(value, list):
        return [expand(item, context) for item in value]
    if isinstance(value, dict):
        return {key: expand(item, context) for key, item in value.items()}
    return value


class Operations:
    """Replaceable executor used by the scheduler; fake executors work without hardware."""

    def __init__(self, callers, nodes):
        self.callers, self.nodes = callers, nodes
        self.lock = threading.RLock()
        self.active_callers = {}
        self.active_episodes = {}

    def stop(self, run_id):
        with self.lock:
            for side in self.active_episodes.get(run_id, set()):
                if self.nodes[side].snapshot()["arm_open"]:
                    self.nodes[side].control.post("/stop", {})
            for caller in self.active_callers.get(run_id, []):
                caller.cancel_playback()
                for side in caller.sides:
                    if self.nodes[side].snapshot()["arm_open"]:
                        self.nodes[side].control.post("/stop", {})

    def execute(self, config, context, cancel, run_id, selection):
        if cancel.is_set():
            return {"cancelled": True}
        config = expand(config, context)
        operation = config["operation"]
        target = config.get("selection", selection)
        if operation == "binding":
            implementation = context["bindings"][config["binding"]]
            result = self.execute({**implementation, "selection": target}, context, cancel, run_id, selection)
            if implementation.get("resultPath"):
                if implementation["operation"] == "script" and result.get("returncode", 0) != 0:
                    raise RuntimeError(f"Script exited with code {result['returncode']}")
                result = lookup(result, implementation["resultPath"])
                if isinstance(result, str):
                    result = json.loads(result)
                if not isinstance(result, dict):
                    raise ValueError("Binding resultPath must select a JSON object")
            return result
        if operation == "dummy":
            cancel.wait(config.get("delaySeconds", 0.25))
            return {**config.get("result", {}), "dummy": True, "note": config.get("note", "Unconfigured adapter")}
        if operation == "finish":
            return {"finish": True}
        if operation in ("emit", "set"):
            return {"name": config["name"], "value": config.get("value", True)}
        if operation == "script":
            import os

            with tempfile.TemporaryFile() as output:
                process = subprocess.Popen(
                    [str(arg) for arg in config["argv"]],
                    cwd=config.get("cwd") or None,
                    env={**os.environ, "ORCHESTRATION_ARMS": target, "ORCHESTRATION_RUN_ID": run_id},
                    stdout=output,
                    stderr=subprocess.STDOUT,
                )
                started = time.monotonic()
                while process.poll() is None:
                    timed_out = config.get("timeoutSeconds") is not None and (
                        time.monotonic() - started >= config["timeoutSeconds"]
                    )
                    if cancel.is_set() or timed_out:
                        process.terminate()
                        process.wait()
                        if timed_out:
                            raise TimeoutError("Configured script time limit reached")
                        return {"cancelled": True}
                    cancel.wait(0.02)
                output.seek(0)
                text = output.read().decode(errors="replace")
                return {"returncode": process.returncode, "stdout": text}
        if operation == "http":
            body = config.get("body")
            request = urllib.request.Request(
                config["url"],
                method=config.get("method", "POST"),
                data=None if body is None else json.dumps(body).encode(),
                headers={
                    "Content-Type": "application/json",
                    "X-Orchestration-Arms": target,
                    **config.get("headers", {}),
                },
            )
            with urllib.request.urlopen(request, timeout=config.get("timeoutSeconds")) as response:
                text = response.read().decode()
                try:
                    data = json.loads(text)
                except json.JSONDecodeError:
                    data = text
                return {"status": response.status, "body": data}
        if operation == "vlm":
            import os
            from dataclasses import asdict

            from dream_hackathon.demo_vlm import LocalVLM, OpenAIVLM

            timeout = config.get("timeoutSeconds")
            if config.get("imagePath"):
                image = pathlib.Path(config["imagePath"]).expanduser().read_bytes()
            else:
                with urllib.request.urlopen(config["imageUrl"], timeout=timeout) as response:
                    image = response.read()
            if config["provider"] == "openai":
                key_name = config.get("apiKeyEnv", "OPENAI_API_KEY")
                key = os.environ.get(key_name)
                if not key:
                    raise ValueError(f"Set {key_name} in the controller environment")
                client = OpenAIVLM(config["model"], key, config.get("url") or "https://api.openai.com/v1", timeout)
            else:
                client = LocalVLM(config["model"], config.get("url") or "http://127.0.0.1:11434", timeout)
            if config.get("schema"):
                result, _raw = client.inspect(image, config["question"], config["schema"])
                return result
            verdict, _raw = client.verify(image, config["question"])
            return asdict(verdict)
        if operation == "episode":
            with self.lock:
                if cancel.is_set():
                    return {"cancelled": True}
                self.active_episodes.setdefault(run_id, set()).add(target)
            try:
                return self.replay(config, cancel, target)
            finally:
                with self.lock:
                    self.active_episodes[run_id].discard(target)
        caller = self.callers[config.get("selection", selection)]
        caller.connect()
        while not cancel.is_set():
            state = caller.snapshot()
            if state.get("observation_error"):
                raise RuntimeError(state["observation_error"])
            if state.get("observation") is not None:
                break
            cancel.wait(0.02)
        with self.lock:
            if cancel.is_set():
                return {"cancelled": True}
            self.active_callers.setdefault(run_id, []).append(caller)
            caller.start(
                {"task": config["task"], "checkpoint": config.get("checkpoint", "dreamlabs/dream-1-beta")},
                run=config.get("run", True),
            )
        started = time.monotonic()
        try:
            while not cancel.is_set():
                state = caller.snapshot()
                if state.get("error"):
                    raise RuntimeError(state["error"])
                if not state["busy"]:
                    return {"calls": state.get("calls"), "rows": state.get("rows")}
                if config.get("run", True) and time.monotonic() - started >= config["durationSeconds"]:
                    self.stop_caller(caller)
                    while caller.snapshot()["busy"] and not cancel.is_set():
                        cancel.wait(0.02)
                    return {"window_complete": True}
                cancel.wait(0.02)
            return {"cancelled": True}
        finally:
            with self.lock:
                self.active_callers[run_id].remove(caller)

    def replay(self, config, cancel, target):
        """Play an existing named target plan through the explicitly targeted arm server."""
        if target == "both":
            raise ValueError("Choose Left or Right for a single-arm recorded episode")
        plan_path = pathlib.Path(config["planPath"]).expanduser()
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        rows = [
            json.loads(line)
            for line in (plan_path.parent / plan["targets_file"]).read_text().splitlines()
            if line.strip()
        ]
        if not rows:
            raise ValueError("Recorded episode has no target rows")
        speed = plan.get("initial_approach_max_joint_speed_rad_s")
        if speed is not None:
            number(speed, "Episode initial approach speed", 0.000001)
            measured = self.nodes[target].snapshot()
            initial_q, initial_grip = measured["q"], measured["grip"]
            duration = max(
                max(abs(a - b) for a, b in zip(rows[0]["q"], initial_q, strict=True)) / speed,
                abs(rows[0]["grip"] - initial_grip) / 1.5,
            )
            approach_started = time.monotonic()
            while not cancel.is_set():
                fraction = min(1, (time.monotonic() - approach_started) / duration) if duration else 1
                self.nodes[target].control.post(
                    "/target",
                    {
                        "q": [a + fraction * (b - a) for a, b in zip(initial_q, rows[0]["q"], strict=True)],
                        "grip": initial_grip + fraction * (rows[0]["grip"] - initial_grip),
                    },
                )
                if fraction == 1:
                    break
                cancel.wait(1 / 30)
        started = time.monotonic()
        for row in rows:
            if cancel.wait(max(0, float(row["t"]) - (time.monotonic() - started))):
                return {"cancelled": True}
            self.nodes[target].control.post("/target", {"q": row["q"], "grip": row["grip"]})
        return {"rows": len(rows), "plan": str(plan_path), "selection": target}

    def stop_caller(self, caller):
        caller.cancel_playback()
        for side in caller.sides:
            if self.nodes[side].snapshot()["arm_open"]:
                self.nodes[side].control.post("/stop", {})


class FlowRun:
    def __init__(self, selection, executor, changed):
        self.selection, self.executor, self.changed = selection, executor, changed
        self.sides = ("left", "right") if selection == "both" else (selection,)
        self.lock = threading.RLock()
        self.cancel = threading.Event()
        self.thread = None
        self.started = None
        self.finished = None
        self.state = {"state": "idle", "busy": False, "error": "", "nodes": {}, "run_id": None}
        self.context = {"signals": {}, "results": {}, "variables": {}}

    def snapshot(self):
        with self.lock:
            return {
                **copy.deepcopy(self.state),
                "context": copy.deepcopy(self.context),
                "elapsedSeconds": max(0, (self.finished or time.monotonic()) - self.started) if self.started else 0,
            }

    def signal(self, name, value):
        if not isinstance(name, str) or not name:
            raise ValueError("Enter a signal name")
        with self.lock:
            self.context["signals"][name] = copy.deepcopy(value)
        self.changed()

    def start(self, graph):
        graph = validate_graph(graph, self.selection, for_run=True)
        with self.lock:
            if self.state["busy"]:
                raise ValueError("Flow is already running or finishing an operation")
            self.cancel = threading.Event()
            self.started = time.monotonic()
            self.finished = None
            self.context = {
                "signals": {},
                "results": {},
                "variables": copy.deepcopy(graph.get("variables", {})),
                "bindings": copy.deepcopy(graph.get("bindings", {})),
            }
            self.state = {"state": "running", "busy": True, "error": "", "nodes": {}, "run_id": uuid.uuid4().hex}
            self.thread = threading.Thread(target=self.run, args=(graph,), daemon=True, name=f"flow:{self.selection}")
            self.thread.start()
        self.changed()

    def stop(self):
        with self.lock:
            if not self.state["busy"]:
                return
            self.cancel.set()
            self.state["state"] = "stopped"
            run_id = self.state["run_id"]
        self.executor.stop(run_id)
        self.changed()

    def status(self, node, status, **values):
        with self.lock:
            previous = self.state["nodes"].get(node, {})
            self.state["nodes"][node] = {**previous, "status": status, **values}
        self.changed()

    def run(self, graph):
        nodes = {node["id"]: node for node in graph["nodes"]}
        outgoing = {}
        incoming = {}
        for edge in graph["edges"]:
            outgoing.setdefault((edge["source"], edge["port"]), []).append(edge)
            incoming.setdefault(edge["target"], []).append(edge)
        ready = []
        waits, futures, joins = [], {}, {}
        pool = ThreadPoolExecutor(max_workers=max(1, len(nodes)), thread_name_prefix="flow-op")

        def token(node, loops=None, groups=(), arrival=None):
            return dict(node=node, loops=dict(loops or {}), groups=groups, arrival=arrival, started=time.monotonic())

        def route(current, port):
            for edge in outgoing.get((current["node"], port), []):
                ready.append(token(edge["target"], current["loops"], current["groups"], edge["id"]))

        ready.append(token(next(node["id"] for node in nodes.values() if node["type"] == "start")))
        try:
            while (ready or waits or futures or joins) and not self.cancel.is_set():
                while ready and not self.cancel.is_set():
                    current = ready.pop(0)
                    node, key = nodes[current["node"]], current["node"]
                    kind, config = node["type"], node["config"]
                    self.status(key, "active")
                    if kind == "action":
                        with self.lock:
                            context = copy.deepcopy(self.context)
                        future = pool.submit(
                            self.executor.execute, config, context, self.cancel, self.state["run_id"], self.selection
                        )
                        futures[future] = current
                        continue
                    if kind == "trigger":
                        waits.append(current)
                        continue
                    port = "next"
                    with self.lock:
                        if kind == "condition":
                            port = (
                                "true"
                                if evaluate(config["condition"], self.context, time.monotonic() - self.started, 0)
                                else "false"
                            )
                        elif kind == "loop":
                            count = current["loops"].get(key, 0)
                            repeat = (
                                count < config.get("count", 1)
                                if config.get("mode", "count") == "count"
                                else (evaluate(config["condition"], self.context, time.monotonic() - self.started, 0))
                            )
                            port = "body" if repeat else "done"
                            if repeat:
                                current["loops"][key] = count + 1
                            else:
                                current["loops"].pop(key, None)
                        elif kind == "fork":
                            current["groups"] += (uuid.uuid4().hex,)
                        elif kind == "join":
                            join_key = (key, current["groups"])
                            bucket = joins.setdefault(join_key, {})
                            bucket[current["arrival"]] = current
                            if len(bucket) < len(incoming.get(key, [])):
                                self.state["nodes"][key].update(status="waiting", arrivals=len(bucket))
                                continue
                            joins.pop(join_key)
                            current["groups"] = current["groups"][:-1]
                    self.status(key, "complete")
                    route(current, port)
                now = time.monotonic()
                for current in list(waits):
                    key = current["node"]
                    config = nodes[key]["config"]
                    with self.lock:
                        matched = evaluate(
                            config["condition"], self.context, now - self.started, now - current["started"]
                        )
                    timeout = config.get("timeoutSeconds")
                    if matched or (timeout is not None and now - current["started"] >= timeout):
                        waits.remove(current)
                        port = "next" if matched else "timeout"
                        self.status(key, "complete" if matched else "timeout")
                        if not matched and not outgoing.get((key, port)):
                            raise TimeoutError(f"{nodes[key]['label']}: configured trigger timeout")
                        route(current, port)
                for future, current in list(futures.items()):
                    if not future.done():
                        continue
                    futures.pop(future)
                    if self.cancel.is_set():
                        continue
                    key = current["node"]
                    config = nodes[key]["config"]
                    operation = config["operation"]
                    if operation == "binding":
                        operation = self.context["bindings"][config["binding"]]["operation"]
                    port = "next"
                    try:
                        result = future.result()
                        if operation == "script" and result.get("returncode", 0) != 0:
                            port = "error"
                        with self.lock:
                            count = self.context["results"].get(key, {}).get("executions", 0) + 1
                            self.context["results"][key] = {
                                **result,
                                "operation": operation,
                                "executions": count,
                            }
                            if result.get("finish") is True and operation == "finish":
                                self.state["state"] = "complete"
                                self.cancel.set()
                            if operation in ("emit", "set"):
                                group = "signals" if operation == "emit" else "variables"
                                self.context[group][result["name"]] = result["value"]
                        if port == "error" and not outgoing.get((key, port)):
                            raise RuntimeError(f"Script exited with code {result['returncode']}")
                    except Exception as error:
                        port = "timeout" if isinstance(error, TimeoutError) else "error"
                        with self.lock:
                            self.context["results"][key] = {
                                **self.context["results"].get(key, {}),
                                "error": str(error),
                                "operation": config["operation"],
                            }
                        if not outgoing.get((key, port)):
                            self.status(key, port, error=str(error))
                            raise
                    self.status(key, "complete" if port == "next" else port)
                    if operation == "finish":
                        self.executor.stop(self.state["run_id"])
                    route(current, port)
                if ready or waits or futures or joins:
                    self.cancel.wait(0.02)
            with self.lock:
                if not self.cancel.is_set():
                    self.state["state"] = "complete"
        except Exception as error:
            with self.lock:
                self.state.update(state="error", error=str(error))
            # No implicit motion or automatic recovery on a graph error.
        finally:
            # Keep the run busy until its already-started operations settle. Stop is explicit.
            pool.shutdown(wait=True)
            with self.lock:
                self.finished = time.monotonic()
                self.state["busy"] = False
            self.changed()


class FlowManager:
    def __init__(self, callers, nodes, publish, path=None, executor=None):
        self.lock, self.publish = threading.RLock(), publish
        self.path = pathlib.Path(path).expanduser() if path is not None else None
        self.graphs = {selection: default_graph() for selection in ("left", "right", "both")}
        from dream_hackathon.demo_strategy import competition_graph

        self.graphs["both"] = competition_graph()
        if self.path is not None and self.path.exists():
            for selection, graph in json.loads(self.path.read_text(encoding="utf-8")).items():
                self.graphs[selection] = validate_graph(graph, selection)
        self.revisions = dict.fromkeys(self.graphs, 0)
        executor = executor or Operations(callers, nodes)
        self.runs = {selection: FlowRun(selection, executor, self.changed) for selection in self.graphs}

    def snapshot(self):
        with self.lock:
            graphs, revisions = copy.deepcopy(self.graphs), dict(self.revisions)
        return {
            selection: {"definition": graph, "revision": revisions[selection], **self.runs[selection].snapshot()}
            for selection, graph in graphs.items()
        }

    def changed(self):
        self.publish({"type": "flows", "flows": self.snapshot()})

    def update(self, selection, graph, revision=None):
        graph = validate_graph(graph, selection)
        with self.lock:
            if revision is not None and revision != self.revisions[selection]:
                raise ValueError("Flow changed in another browser; reload its saved version before saving")
            self.graphs[selection] = graph
            self.revisions[selection] += 1
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.write_text(json.dumps(self.graphs, indent=2), encoding="utf-8")
        self.changed()

    def start(self, selection):
        with self.lock:
            graph = copy.deepcopy(self.graphs[selection])
        self.runs[selection].start(graph)

    def stop_sides(self, sides):
        for run in self.runs.values():
            if set(sides).intersection(run.sides):
                run.stop()

    def close(self):
        self.stop_sides(["left", "right"])
