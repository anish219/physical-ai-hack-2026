# Running your model

For the browser-based single-arm API caller, see [Independent arm Dream-1 caller](dream_caller.md).
It uses the independent arm server, live center/selected-wrist cameras and dummy data for the opposite arm.

Your fine-tuned model runs on the arms with `dreamcli`, the Dream library's robot runner: it reads the arms and cameras, asks the model for the next moves and drives the arms, with its own speed and safety limits. It runs from `~/dream-env` on the station's computer. Close `dhk-collect` first: only one program can use the arms at a time.

The collector dashboard's arm selection applies to Oculus teleoperation only. The independent API caller above
provides single-arm inference; the existing upload/fine-tuning tools and separately installed `dreamcli` are unchanged.
The selector does not change that external runner. For code using this kit's arm adapters, `command(joints, grips)`
accepts just `left`, just `right`, or both dictionary entries, and sends no target to an omitted arm. This allows
independent command routing. Use `station_lock.acquire_sides` and a selected-arm config for separate processes
(see [collection controls](data_collection.md#separate-controller-processes)). The standard Dream runner and
fine-tuning tools remain bimanual: use the collector's `/both` dataset with them. Single-arm recordings under
`/left` or `/right` need a matching custom embodiment and runner.

## 1. The station's dreamcli setup

The organizers have already run `dreamcli setup` on your station: it saved the arms and cameras in `~/.dreamlabs/station.yaml`, each camera in the same role as in your recordings. The model learned which view is which, so a swapped camera would make it act on the wrong image. Do not move the cameras. Turn the Dream library's environment on before the commands below:

```bash
source ~/dream-env/bin/activate
```

## 2. Try it without moving the arms

```bash
# YAM: your fine-tune drives the joints it learned
dreamcli robot run --checkpoint dream-1-ft:<account>-myteam-cubes --task "put the red cube in the bowl" --action-space joint --dry-run
# Metal or Maker arm: your fine-tune answers tool poses, which dreamcli turns into the arm's joints
dreamcli robot run --checkpoint dream-1-ft:<account>-myteam-cubes --task "put the red cube in the bowl" --dry-run
```

A dry run calls the model for real and writes nothing to the arms. The first call to a new model waits while the model starts, which can take up to an hour; later calls answer in well under a second. `--task` is the instruction the model gets: describe the task in plain words, and if you recorded with `--task`, use that same text.

## 3. Run it

With a hand on the emergency stop, run the same command without `--dry-run`. A run starts by moving both arms to the home pose in the station's config.

To start runs from the browser and watch the cameras, use Live View instead: `dreamcli robot up` (on a YAM fine-tune, `dreamcli robot up joint`), then press Start on the platform's Live View page.

## No fine-tune yet

Dream-1's base model runs on YAM, Metal and Maker arms without fine-tuning. Leave out `--checkpoint` (and `--action-space joint`) to run it: `dreamcli robot run --task "..." --dry-run`. It is a good way to check the station while your job trains, and a fallback if it is not done in time.

The Dream docs cover all of this in more depth: [Getting started](https://platform.dreamlabs.ai/api-documentation/), and [Fine-tuning](https://platform.dreamlabs.ai/api-documentation/fine-tuning/), step 8.
