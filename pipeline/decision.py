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
        messages = []
        seen_this_frame = set()

        for det in detections:
            label = det["label"]
            distance = det["distance"]

            if distance not in ("Close", "Medium"):
                continue

            key = f"{label}_{distance}"
            if key in seen_this_frame:
                continue
            seen_this_frame.add(key)

            allowed, count = self._can_speak(key, now)

            if allowed:
                self.last_spoken[key] = (now, count + 1, now)
                if distance == "Close":
                    messages.append(f"{label} very close")
                else:
                    messages.append(f"{label} ahead, caution")
            else:
                last_time, c, _ = self.last_spoken[key]
                self.last_spoken[key] = (last_time, c, now)

        return messages