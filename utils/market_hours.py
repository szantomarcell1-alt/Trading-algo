"""
Market Hours & Session Filter Module
Ensures scalping occurs only during high-liquidity, tight-spread market sessions.
"""

from datetime import datetime, timezone
from typing import Tuple


class MarketHoursGuard:
    def __init__(
        self,
        allowed_start_utc_hour: int = 7,
        allowed_end_utc_hour: int = 20,
        avoid_rollover_window: bool = True,
        avoid_friday_close: bool = True,
    ):
        self.allowed_start_utc_hour = allowed_start_utc_hour
        self.allowed_end_utc_hour = allowed_end_utc_hour
        self.avoid_rollover_window = avoid_rollover_window
        self.avoid_friday_close = avoid_friday_close

    def is_trading_allowed(self, dt: datetime = None) -> Tuple[bool, str]:
        """
        Validates if trading is allowed at the given datetime (UTC).
        Returns: (allowed: bool, reason: str)
        """
        if dt is None:
            dt = datetime.now(timezone.utc)
        elif dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        weekday = dt.weekday()  # Monday = 0, Sunday = 6
        hour = dt.hour
        minute = dt.minute

        # 1. Weekend check (Forex / CFD closed from Friday late night to Sunday night)
        if weekday == 5:  # Saturday
            return False, "Piac zárva: Szombat"
        if weekday == 6 and hour < 21:  # Sunday before 21:00 UTC
            return False, "Piac zárva: Vasárnap este előtti időszak"

        # 2. Friday close protection (avoid weekend gap risk)
        if self.avoid_friday_close and weekday == 4:  # Friday
            if hour >= 19:
                return False, "Pénteki zárás előtti stop: Hétvégi gap kockázat kerülése"

        # 3. Rollover window check (21:00 - 23:00 UTC / GMT)
        # Brokerek ekkor cserélik a swapokat, a liquidity kiszárad, a spreadek 5-10x-re tágulnak!
        if self.avoid_rollover_window:
            if 21 <= hour < 23:
                return False, f"Rollover ablak ({hour:02d}:{minute:02d} UTC): Táguló spread és likviditáshiány"

        # 4. Configured session window (e.g. London & NY sessions: 07:00 - 20:00 UTC)
        if hour < self.allowed_start_utc_hour or hour >= self.allowed_end_utc_hour:
            return (
                False,
                f"Kereskedési időablakon kívül ({hour:02d}:{minute:02d} UTC). "
                f"Megengedett sáv: {self.allowed_start_utc_hour:02d}:00 - {self.allowed_end_utc_hour:02d}:00 UTC",
            )

        return True, "Piaci likviditás aktív (Kereskedés engedélyezett)"

    def get_current_session_name(self, dt: datetime = None) -> str:
        if dt is None:
            dt = datetime.now(timezone.utc)
        hour = dt.hour

        # Session overlaps
        if 8 <= hour < 12:
            return "London Session (Nagy likviditás)"
        elif 12 <= hour < 16:
            return "London / New York Átfedés (CSÚCS Likviditás)"
        elif 16 <= hour < 20:
            return "New York Session (Magas volatilitás)"
        elif 0 <= hour < 8:
            return "Ázsiai / Tokiói Session (Alacsonyabb volatilitás)"
        else:
            return "Csendes / Rollover időszak"
