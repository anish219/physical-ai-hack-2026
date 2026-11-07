"""Explicit motor torque-off for both station arms, without homing or enabling."""


def kill_station_arms(station, active=None) -> list[str]:
    from lerobot.robots.metal_follower import MetalFollower, MetalFollowerConfig
    from makermodslab import torque

    problems = []
    handled = set()
    if active is not None:
        handled.update(active._devices)
        try:
            problems.extend(active.power_off())
        except Exception as error:
            problems.append(f"Active arm torque disable failed: {error}")
        finally:
            try:
                active.close()
            except Exception as error:
                problems.append(f"Active arm disconnect failed: {error}")
    for side, arm in station.arms.items():
        if side in handled:
            continue
        try:
            device = MetalFollower(MetalFollowerConfig(
                port=arm.port, id=f"dream_hackathon_{side}", can_interface=station.makermods.can_interface,
            ))
            # Open without a handshake, disable, then close without enabling.
            problems.extend(torque.de_energize_can_bus(device.bus, f"{side} follower arm"))
        except Exception as error:
            problems.append(f"{side} torque disable failed: {error}")
    return problems
