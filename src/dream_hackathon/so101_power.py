"""Explicit motor torque-off for the controlled arm, without homing or enabling."""


def kill_controlled_arm(station, side: str, active=None) -> list[str]:
    from lerobot.robots.metal_follower import MetalFollower, MetalFollowerConfig
    from makermodslab import metal_gripper, torque

    problems = []
    if active is not None:
        try:
            problems.extend(active.power_off())
        except Exception as error:
            problems.append(f"Active arm torque disable failed: {error}")
        finally:
            try:
                active.close()
            except Exception as error:
                problems.append(f"Active arm disconnect failed: {error}")
        return problems
    try:
        device = MetalFollower(
            MetalFollowerConfig(
                port=station.arms[side].port,
                id=f"dream_hackathon_{side}",
                can_interface=station.makermods.can_interface,
            )
        )
        metal_gripper.install_metal_gripper(device, "")
        # Open only the selected bus without a handshake; disable, then close.
        problems.extend(torque.de_energize_can_bus(device.bus, f"{side} follower arm"))
    except Exception as error:
        problems.append(f"{side} torque disable failed: {error}")
    return problems
