"""One block to a free table spot with the night loop's code (recorded, every check on): untangling a pile by hand.

    FT_ROOT=night_nov8 python -m calib.night_move COLOUR#N[:L1][@X_CM,Y_CM[,YAW_DEG]]  # :L1 on a hidden block; @ where it is
"""
from __future__ import annotations

import math
import sys
import time

from hillclimb.runner.calib import data_loop as D
from hillclimb.runner.calib import tall_tower as T


def main():
    arg, _, at = sys.argv[1].partition("@")
    bid, _, lvl = arg.partition(":L")
    D.take_token()
    t = T.Tall(T.TALL / time.strftime("%m%d-%H%M%S"))
    t.op.st["target"] = None
    D.rec("/start")
    clean = True
    try:
        t.look_all("look-move")
        if lvl:  # on a block hidden under it: its real level, so the pick goes to its height
            t.belief[bid].update(level=int(lvl), z0=T.TABLE_Z + int(lvl) * T.H)
        if at:  # where it is (two touching blocks of one colour fit as one blob between them)
            v = [float(u) for u in at.split(",")]
            t.belief[bid].update(x=v[0] / 100, y=v[1] / 100, t_seen=time.time())
            if len(v) > 2:
                t.belief[bid]["yaw"] = math.radians(v[2])  # trusted: no re-look onto the merged blob
        t.put_blocks()
        x, y, yaw = t.pick_spot(bid, [0])
        t.cyc = {"tower_order": [], "fallbacks": [], "cycle": 0, "untangle": bid}
        t.block(bid, (x, y, yaw, 0, None), 0, 1, "untangle")
    except BaseException as e:  # noqa: BLE001
        D.status(f"night move {bid} stopped: {type(e).__name__}: {e}")
        st = O_state = T.arm_state()
        clean = st["grip"] > 0.75 and not st.get("grip_holding")
    finally:
        T.finish(t, clean)


if __name__ == "__main__":
    main()
