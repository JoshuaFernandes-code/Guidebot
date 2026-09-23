import time
from config import CONFIG

class DecisionEngine:
    def __init__(self):
        self.last_spoken = {}
        self.reset_after_seconds = 6

    def _can_speak(self, key, now):
        if key not in self.last_spoken:
            return True, 0
        last_time, count, last_seen = self.last_spoken[key]
        if now - last_seen > self.reset_after_seconds:
            return True, 0
        if count >= CONFIG["max_repeats"]:
            return False, count
        if now - last_time < CONFIG["warning_cooldown_seconds"]:
            return False, count
        return True, count

    def decide(self, detections):
        now = time.time()
        danger_classes = CONFIG["danger_classes"]
        messages = []

        for det in detections:
            label = det["label"]
            distance = det["distance"]
            is_danger = label in danger_classes

            if not is_danger or distance != "Close":
                continue

            key = f"{label}_{distance}"
            allowed, count = self._can_speak(key, now)

            if allowed:
                self.last_spoken[key] = (now, count + 1, now)
                messages.append(f"warning, {label} very close")

        return messages

    def get_direction(self, x1, x2, frame_width):
        center_x = (x1 + x2) / 2
        third = frame_width / 3
        if center_x < third:
            return "left"
        elif center_x > third * 2:
            return "right"
        else:
            return "ahead"

    def answer_query(self, query, detections, frame_width):
        matches = [d for d in detections if d["direction"] == query]

        if not matches:
            return f"nothing detected to your {query}" if query != "ahead" else "nothing detected ahead"

        closest = matches[0]
        if query != "ahead":
            return f"{closest['label']} is {closest['distance'].lower()} to your {query}"
        else:
            return f"{closest['label']} is {closest['distance'].lower()} ahead"