"""Register named sequence-monitor scripts and the VLM prompt evaluator.

SequenceManager calls register_all() once at startup. Pointers registered here
appear in each subphase's Monitor → Script pointer dropdown. Teammate agents:
read docs/sequence_monitors.md before wiring a model or a new script.
"""


def register_all(registry):
    """Attach Anish's monitors to the shared registry.

    registry.set_prompt_evaluator(evaluate)
        evaluate(prompt: str, context: dict) -> "yes"|"no"|"uncertain"|dict
        Called after a subphase whose monitor.type is "prompt".

    registry.register_script(pointer, evaluate, label=None)
        evaluate(context: dict) -> "yes"|"no"|"uncertain"|dict
        pointer is the value stored on the step and shown in the editor.

    Context always includes:
        selection: "left"|"right"|"both"
        sides: tuple of arm sides this sequence owns
        step: the subphase dict (includes monitor)
        arms: {side: arm snapshot}
        camera_urls: {side: "http://127.0.0.1:<arm-port>"} for live JPEGs

    Return {"result": "yes"|"no"|"uncertain", "detail": "..."}.
    Results are recorded on the run; they do not yet gate the next subphase.

    Example:
        def usb_unplugged(context):
            return {"result": "uncertain", "detail": "wire the vision check"}

        def vlm_prompt(prompt, context):
            return {"result": "unwired", "detail": prompt}

        registry.register_script("usb_unplugged", usb_unplugged, label="USB unplugged")
        registry.set_prompt_evaluator(vlm_prompt)
    """
    return registry
