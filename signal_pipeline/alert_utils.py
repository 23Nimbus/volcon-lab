# Utility functions for alert level calculations

def classify_alert_level(score):
    """Return the alert level emoji string for a given score.

    The score may be fractional. It is rounded to the nearest integer
    before classification so that small adjustments do not lead to
    unexpected alert levels.
    """
    rounded = round(score)
    if rounded <= 1:
        return "🟢 Watch"
    elif rounded == 2:
        return "🟡 Tension"
    else:
        return "🔴 Breakout Potential"
