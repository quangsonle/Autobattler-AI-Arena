import math
import numpy as np
from dataclasses import dataclass
from typing import List
from config import (
    MAP_WIDTH, MAP_HEIGHT, A_ZONE_Y, DEAD_ZONE_Y, B_ZONE_Y,
    PLAYER_SPEED, BULLET_SPEED, FIRE_COOLDOWN, HITBOX_RADIUS
)

@dataclass
class Bullet:
    x: float
    y: float
    vx: float
    vy: float
    owner: str  # 'A' or 'B'

    def update(self) -> bool:
        self.x += self.vx
        self.y += self.vy
        return (0 <= self.x < MAP_WIDTH) and (0 <= self.y < MAP_HEIGHT)

@dataclass
class Player:
    id: str
    x: float
    y: float
    y_min: float
    y_max: float
    cooldown: int = 0
    score: int = 0

    def move(self, move_act: int):
        # 0:Idle, 1:Left, 2:Right, 3:Advance, 4:Retreat
        dx, dy = 0.0, 0.0
        if move_act == 1: dx = -1.0
        elif move_act == 2: dx = 1.0

        if self.id == 'A':
            if move_act == 3: dy = 1.0    # Advance (Down)
            elif move_act == 4: dy = -1.0 # Retreat (Up)
        else:
            if move_act == 3: dy = -1.0   # Advance (Up)
            elif move_act == 4: dy = 1.0    # Retreat (Down)

        self.x = max(0.0, min(MAP_WIDTH - 1.0, self.x + dx * PLAYER_SPEED))
        self.y = max(self.y_min, min(self.y_max, self.y + dy * PLAYER_SPEED))

    def tick_cooldown(self):
        if self.cooldown > 0:
            self.cooldown -= 1

class GameEngine:
    def __init__(self):
        self.reset()

    def reset(self):
        self.player_a = Player('A', 25.0, 20.0, A_ZONE_Y[0], A_ZONE_Y[1])
        self.player_b = Player('B', 25.0, 280.0, B_ZONE_Y[0], B_ZONE_Y[1])
        self.bullets: List[Bullet] = []
        self.last_hit: str = None
        self.ticks = 0

    def auto_fire(self, player: Player):
        if player.cooldown > 0:
            return

        direction_y = 1.0 if player.id == 'A' else -1.0
        vx = 0.0  # ALWAYS STRAIGHT!
        vy = direction_y * BULLET_SPEED

        self.bullets.append(Bullet(player.x, player.y, vx, vy, player.id))
        player.cooldown = FIRE_COOLDOWN

    def step(self, move_a: int, move_b: int):
        self.ticks += 1
        self.last_hit = None

        self.player_a.tick_cooldown()
        self.player_b.tick_cooldown()

        # Update position
        self.player_a.move(move_a)
        self.player_b.move(move_b)

        # Auto-fire straight on cooldown
        self.auto_fire(self.player_a)
        self.auto_fire(self.player_b)

        surviving_bullets = []
        for b in self.bullets:
            if not b.update():
                continue

            if b.owner == 'B':
                if math.hypot(b.x - self.player_a.x, b.y - self.player_a.y) < HITBOX_RADIUS:
                    self.player_b.score += 1
                    self.last_hit = 'A'
                    continue

            if b.owner == 'A':
                if math.hypot(b.x - self.player_b.x, b.y - self.player_b.y) < HITBOX_RADIUS:
                    self.player_a.score += 1
                    self.last_hit = 'B'
                    continue

            surviving_bullets.append(b)

        self.bullets = surviving_bullets

    def get_canonical_features(self, player_id: str) -> np.ndarray:
        is_a = (player_id == 'A')
        self_p = self.player_a if is_a else self.player_b
        rival_p = self.player_b if is_a else self.player_a

        def to_canonical(x, y, vx, vy):
            if is_a:
                return x, y, vx, vy
            else:
                return x, (MAP_HEIGHT - 1.0 - y), vx, -vy

        sx, sy, _, _ = to_canonical(self_p.x, self_p.y, 0, 0)
        rx, ry, _, _ = to_canonical(rival_p.x, rival_p.y, 0, 0)

        features = [
            sx / MAP_WIDTH,
            sy / 100.0,
            self_p.cooldown / float(FIRE_COOLDOWN)
        ]

        features.extend([
            (rx - sx) / MAP_WIDTH,
            (ry - sy) / MAP_HEIGHT
        ])

        enemy_bullets = []
        own_bullets = []
        for b in self.bullets:
            bx, by, bvx, bvy = to_canonical(b.x, b.y, b.vx, b.vy)
            if b.owner != player_id:
                dist = math.hypot(bx - sx, by - sy)
                enemy_bullets.append((dist, bx, by, bvx, bvy))
            else:
                dist = math.hypot(bx - rx, by - ry)
                own_bullets.append((dist, bx, by))

        enemy_bullets.sort(key=lambda item: item[0])
        for i in range(5):
            if i < len(enemy_bullets):
                _, bx, by, bvx, bvy = enemy_bullets[i]
                time_to_hit = (by - sy) / (-bvy) if bvy < -1e-4 else 10.0
                time_to_hit = max(0.0, min(5.0, time_to_hit)) / 5.0
                features.extend([
                    (bx - sx) / MAP_WIDTH,
                    (by - sy) / 100.0,
                    bvx / BULLET_SPEED,
                    bvy / BULLET_SPEED,
                    time_to_hit,
                    1.0
                ])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        own_bullets.sort(key=lambda item: item[0])
        for i in range(3):
            if i < len(own_bullets):
                _, bx, by = own_bullets[i]
                features.extend([
                    (bx - rx) / MAP_WIDTH,
                    (by - ry) / MAP_HEIGHT,
                    1.0
                ])
            else:
                features.extend([0.0, 0.0, 0.0])

        return np.array(features, dtype=np.float32)
