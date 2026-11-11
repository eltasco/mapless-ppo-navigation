"""Interactive Pygame rollout inspector, loaded only when requested."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from .experiments import _check_spaces, environment, require_checkpoint
from .scenarios import parse_radius, set_episode_radius

BG = (21, 25, 33)
PANEL_BG = (32, 38, 49)
WHITE = (235, 238, 245)
GRAY = (150, 158, 170)
GREEN = (75, 210, 150)
RED = (235, 90, 95)
BLUE = (65, 145, 245)
YELLOW = (245, 195, 75)
CYAN = (70, 205, 225)
WIDTH, HEIGHT, MAP_WIDTH = 1200, 800, 880


def _camera(robot):
    coords_x = [0.0, robot.x, robot.goal_x] + [x for x, _ in robot.obstacles]
    coords_y = [0.0, robot.y, robot.goal_y] + [y for _, y in robot.obstacles]
    min_x, max_x = min(coords_x) - 1.0, max(coords_x) + 1.0
    min_y, max_y = min(coords_y) - 1.0, max(coords_y) + 1.0
    scale = min((MAP_WIDTH - 80) / (max_x - min_x), (HEIGHT - 80) / (max_y - min_y), 160)
    return (min_x + max_x) / 2, (min_y + max_y) / 2, scale


def _xy(x, y, cam):
    cx, cy, scale = cam
    return int(MAP_WIDTH / 2 + (x - cx) * scale), int(HEIGHT / 2 - (y - cy) * scale)


def view(checkpoint: str | Path, stage: str, obstacles: int, *, radius_spec="original", fps=30, seed_start=40000):
    # Optional dependency is intentionally isolated from headless evaluation.
    import pygame

    require_checkpoint(checkpoint)
    model = PPO.load(str(checkpoint), device="cpu")
    env = environment(obstacles, stage)
    _check_spaces(model, env)
    radius_config = parse_radius(radius_spec)
    pygame.init()
    try:
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Mapless PPO Navigation | Rollout Inspector")
        font = pygame.font.SysFont("consolas", 22)
        small = pygame.font.SysFont("consolas", 17)
        clock = pygame.time.Clock()

        def draw_text(message, x, y, color=WHITE, large=False):
            screen.blit((font if large else small).render(str(message), True, color), (x, y))

        def reset_episode(seed):
            r = set_episode_radius(env, radius_config, seed)
            obs, info = env.reset(seed=seed)
            if not math.isclose(r, env.robot.obstacle_radius, abs_tol=1e-9):
                raise RuntimeError("Obstacle radius reset unexpectedly")
            return obs, info, _camera(env.robot), [(env.robot.x, env.robot.y)]

        seed = seed_start
        episode_number = 1
        observation, info, camera, path = reset_episode(seed)
        done = paused = False
        show_lidar = True
        reward_total = speed = omega = 0.0
        running = True
        while running:
            clock.tick(fps)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_l:
                        show_lidar = not show_lidar
                    elif event.key in (pygame.K_n, pygame.K_r):
                        if event.key == pygame.K_n:
                            seed += 1
                            episode_number += 1
                        observation, info, camera, path = reset_episode(seed)
                        done = paused = False
                        reward_total = speed = omega = 0.0
                    elif event.key == pygame.K_s:
                        folder = Path("results/screenshots")
                        folder.mkdir(parents=True, exist_ok=True)
                        filename = folder / f"rollout_{episode_number:03d}_{stage}_{obstacles}obs.png"
                        pygame.image.save(screen, filename)
                        print(f"Screenshot: {filename}")
            if not running:
                break

            robot = env.robot
            if not done and not paused:
                action, _ = model.predict(observation, deterministic=True)
                speed = float(np.clip(action[0], -1, 1)) * robot.max_linear_speed
                omega = float(np.clip(action[1], -1, 1)) * robot.max_angular_speed
                observation, reward, terminated, truncated, info = env.step(action)
                reward_total += float(reward)
                path.append((robot.x, robot.y))
                done = bool(terminated or truncated)
                if done:
                    outcome = (
                        "SUCCESS" if info.get("goal_reached", False)
                        else "COLLISION" if info.get("collision", False)
                        else "TIMEOUT"
                    )
                    print(f"Episode {episode_number}: {outcome}, steps={robot.steps}, reward={reward_total:.2f}")

            screen.fill(BG)
            scale = camera[2]
            robot_pos = _xy(robot.x, robot.y, camera)

            # 1-metre grid.
            for i in range(-15, 16):
                pygame.draw.line(screen, (42, 48, 58), _xy(i, -15, camera), _xy(i, 15, camera), 1)
                pygame.draw.line(screen, (42, 48, 58), _xy(-15, i, camera), _xy(15, i, camera), 1)

            if len(path) > 1:
                pygame.draw.lines(screen, CYAN, False, [_xy(x, y, camera) for x, y in path], 2)

            lidar = observation[-len(robot.sensor_angles):]
            if show_lidar:
                for angle, distance in zip(robot.sensor_angles, lidar):
                    absolute_angle = robot.theta + angle
                    endpoint = _xy(
                        robot.x + float(distance) * math.cos(absolute_angle),
                        robot.y + float(distance) * math.sin(absolute_angle), camera,
                    )
                    pygame.draw.line(screen, (65, 115, 110), robot_pos, endpoint, 1)
                    if float(distance) < robot.sensor_range - 0.01:
                        pygame.draw.circle(screen, GREEN, endpoint, 3)

            for ox, oy in robot.obstacles:
                pos = _xy(ox, oy, camera)
                radius_px = max(1, int(robot.obstacle_radius * scale))
                pygame.draw.circle(screen, (145, 67, 58), pos, radius_px)
                pygame.draw.circle(screen, RED, pos, radius_px, 2)

            pygame.draw.circle(
                screen, (45, 115, 75), _xy(robot.goal_x, robot.goal_y, camera),
                max(1, int(robot.goal_radius * scale)),
            )
            pygame.draw.circle(screen, GREEN, _xy(robot.goal_x, robot.goal_y, camera), 3)
            r = max(1, int(robot.robot_radius * scale))
            pygame.draw.circle(screen, BLUE, robot_pos, r)
            pygame.draw.circle(screen, WHITE, robot_pos, r, 2)
            front = (
                robot_pos[0] + int((r + 17) * math.cos(robot.theta)),
                robot_pos[1] - int((r + 17) * math.sin(robot.theta)),
            )
            pygame.draw.line(screen, WHITE, robot_pos, front, 4)
            pygame.draw.rect(screen, PANEL_BG, (MAP_WIDTH, 0, WIDTH - MAP_WIDTH, HEIGHT))

            x = MAP_WIDTH + 20
            draw_text("PPO / MAPLESS NAVIGATION", x, 22, CYAN, large=True)
            draw_text(Path(checkpoint).stem[:31], x, 57, GRAY)
            draw_text(f"Scenario: {stage} / {len(robot.obstacles)} obstacles", x, 100)
            draw_text(f"Episode: {episode_number}  step: {robot.steps}/{robot.max_steps}", x, 130)
            distance = math.hypot(robot.goal_x - robot.x, robot.goal_y - robot.y)
            clearance = min(
                math.hypot(robot.x - ox, robot.y - oy) - robot.robot_radius - robot.obstacle_radius
                for ox, oy in robot.obstacles
            )
            draw_text(f"Goal distance: {distance:.3f} m", x, 190)
            draw_text(f"Linear speed: {speed:+.3f} m/s", x, 218)
            draw_text(f"Yaw rate: {omega:+.3f} rad/s", x, 246)
            draw_text(f"Nearest LiDAR: {min(lidar):.3f} m", x, 288)
            draw_text(f"Clearance: {clearance:.3f} m", x, 316, RED if clearance < 0.1 else WHITE)
            draw_text(f"Obstacle radius: {robot.obstacle_radius:.2f} m", x, 345)
            if len(robot.obstacles) >= 2:
                gaps = [
                    math.hypot(x1-x2, y1-y2) - 2*robot.obstacle_radius
                    for i, (x1,y1) in enumerate(robot.obstacles)
                    for x2,y2 in robot.obstacles[i+1:]
                ]
                draw_text(f"Minimum pair gap: {min(gaps):.3f} m", x, 385)
            draw_text(f"Episode reward: {reward_total:.2f}", x, 428)
            outcome = (
                "SUCCESS" if info.get("goal_reached") else
                "COLLISION" if info.get("collision") else "TIMEOUT"
            ) if done else "PAUSED" if paused else "RUNNING"
            color = GREEN if outcome == "SUCCESS" else RED if outcome == "COLLISION" else YELLOW if done or paused else CYAN
            draw_text(outcome, x, 470, color, large=True)
            draw_text("CONTROLS", x, 585, CYAN)
            for i, label in enumerate((
                "N  next episode", "R  replay episode", "SPACE  pause/resume",
                "L  toggle LiDAR", "S  save screenshot", "ESC  exit",
            )):
                draw_text(label, x, 615 + 27 * i)
            pygame.display.flip()
    finally:
        env.close()
        pygame.quit()
