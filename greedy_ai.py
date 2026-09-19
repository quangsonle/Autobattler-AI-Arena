import math
from config import MAP_WIDTH

class GreedyAI:
    def __init__(self, player_id: str):
        self.player_id = player_id
        self.strafe_dir = 1

    def get_action(self, engine):
        is_a = (self.player_id == 'A')
        me = engine.player_a if is_a else engine.player_b
        rival = engine.player_b if is_a else engine.player_a

        # 1. Predictive Bullet Dodging
        incoming = []
        for b in engine.bullets:
            if b.owner == self.player_id:
                continue

            approaching = (b.vy > 0) if is_a else (b.vy < 0)
            if approaching:
                time_to_y = (me.y - b.y) / b.vy
                if time_to_y > 0:
                    proj_x = b.x + b.vx * time_to_y
                    dist_y = abs(me.y - b.y)
                    incoming.append((time_to_y, dist_y, proj_x, b))

        incoming.sort(key=lambda item: item[0])
        move_act = 0
        evading = False

        for time_to_y, dist_y, proj_x, b in incoming:
            if abs(proj_x - me.x) < 5.0 and dist_y < 160.0:
                evading = True
                if proj_x >= me.x:
                    move_act = 1  # Evade Left
                else:
                    move_act = 2  # Evade Right

                if me.x < 7.0: move_act = 2
                elif me.x > MAP_WIDTH - 7.0: move_act = 1
                break

        # 2. Dynamic Strafing & Rival Tracking
        if not evading:
            dx = rival.x - me.x
            if abs(dx) > 3.0:
                move_act = 1 if dx < 0 else 2
            else:
                if me.x < 15.0: self.strafe_dir = 1
                elif me.x > MAP_WIDTH - 15.0: self.strafe_dir = -1
                move_act = 2 if self.strafe_dir == 1 else 1

        return move_act
