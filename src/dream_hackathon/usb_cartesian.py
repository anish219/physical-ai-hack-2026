"""Frame-by-frame image feedback for lateral XY while descending in world Z."""

import time

import numpy as np

from dream_hackathon.bundle_pull import move_to


def calibrate(arm, vision, kinematics, settings, grip, report):
    """Measure image response to world X/Y/Z translations at the aligned pose."""
    q, _ = arm.state()
    origin = kinematics.fk(q)
    positions, features = [], []
    for axis in range(3):
        samples, measured = [], []
        for sign in (1, -1):
            pose = origin.copy()
            pose[axis, 3] += sign * settings['probe_m']
            target = kinematics.solve(pose, arm.state()[0])
            move_to(arm, target, grip, settings['motion_rad_s'],
                    hz=settings['hz'], settle=settings['settle_s'])
            feature, status, image = vision.measure(arm.image())
            samples.append(feature)
            measured.append(kinematics.fk(arm.state()[0])[:3, 3])
            report('cartesian_calibration', {'axis': axis, 'sign': sign, **status}, image)
        positions.append(measured[0] - measured[1])
        features.append(samples[0] - samples[1])
        move_to(arm, q, grip, settings['motion_rad_s'], hz=settings['hz'], settle=settings['settle_s'])
    displacement = np.column_stack(positions)
    if np.linalg.matrix_rank(displacement) != 3:
        raise ValueError('Could not measure three independent world translation probes')
    jacobian = np.column_stack(features) @ np.linalg.inv(displacement)
    if np.linalg.matrix_rank(jacobian) != 3:
        raise ValueError('Image geometry does not resolve world X/Y separately from insertion depth')
    return jacobian


def lateral_correction(jacobian, image_error, gain, step_m):
    """Separate lateral error from the image motion caused by descending.

    The third estimated displacement accounts for depth change. Only X and Y
    are fed back; Z remains the explicitly commanded world-frame down stroke.
    A fixed 2-D pixel setpoint would otherwise fight perspective/depth motion.
    """
    displacement = np.linalg.lstsq(jacobian, image_error, rcond=None)[0]
    xy = gain * displacement[:2]
    xy *= min(1, step_m / max(float(np.linalg.norm(xy)), 1e-12))
    return xy


def descend(arm, vision, kinematics, settings, grip, report, *,
            jacobian=None, clock=time.monotonic, sleep=time.sleep):
    """Keep world orientation; take a fresh image and correct X/Y for every target."""
    if jacobian is None:
        jacobian = calibrate(arm, vision, kinematics, settings, grip, report)
    q, _ = arm.state()
    origin = kinematics.fk(q)
    final_z = origin[2, 3] - settings['distance_m']
    feature, _status, _image = vision.measure(arm.image())
    previous_features, previous_position = feature, origin[:3, 3].copy()
    previous_time = clock()
    commanded_z = origin[2, 3]
    frames = 0
    while True:
        feature, status, image = vision.measure(arm.image())
        measured_q, _ = arm.state()
        measured = kinematics.fk(measured_q)
        now = clock()
        dt = now - previous_time
        dx = measured[:3, 3] - previous_position
        df = feature - previous_features
        if dx @ dx > 1e-8:
            jacobian += settings.get('adapt_gain', 0.1) * np.outer(df - jacobian @ dx, dx) / (dx @ dx)
        xy = lateral_correction(jacobian, vision.target - feature,
                                settings['lateral_gain'], settings['lateral_step_m'])
        commanded_z = max(final_z, commanded_z - settings['speed_m_s'] * dt)
        target = origin.copy()  # Preserves the aligned world-frame orientation.
        target[:2, 3] = measured[:2, 3] + xy
        target[2, 3] = commanded_z
        q_target = kinematics.solve(target, measured_q)
        arm.command(q_target, grip)
        frames += 1
        report('descending', {**status, 'frame': frames, 'world_position_m': measured[:3, 3].tolist(),
                              'target_world_position_m': target[:3, 3].tolist(),
                              'xy_correction_m': xy.tolist(), 'q_target': q_target.tolist()}, image)
        previous_time, previous_position, previous_features = now, measured[:3, 3].copy(), feature
        if commanded_z == final_z and abs(measured[2, 3] - final_z) <= settings['depth_tolerance_m']:
            return {'descent_complete': True, 'frames': frames, 'world_down_m': settings['distance_m'],
                    'final_world_position_m': measured[:3, 3].tolist()}
        sleep(max(0, 1 / settings['hz'] - (clock() - now)))
