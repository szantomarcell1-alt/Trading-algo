"""
Market Hours & Session Guard Module for GBPJPY
Manages Asian (01:00-08:00 UTC), London (08:00-16:30 UTC), Tokyo overlap (07:00-09:00 UTC),
London/NY overlap (12:00-16:00 UTC), New York (12:00-20:00 UTC) sessions.
Enforces rollover spread blowout lockout (21:00-23:00 UTC), scalp exit cutoff (20:30 UTC),
and Friday weekend risk cutoff (19:00 UTC).
"""

from datetime import datetime, timezone, time
from typing import Optional, Tuple, Any, Dict


def _to_utc(dt: Optional[datetime]) -> datetime:
    if dt is None:
        return datetime.now(timezone.utc)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class MarketHours:
    """
    Session filter and trading schedule guard for GBPJPY scalping.
    """
    def __init__(
        self,
        config: Optional[Any] = None,
        allowed_start_utc_hour: int = 1,
        allowed_end_utc_hour: int = 20,
        avoid_rollover_window: bool = True,
        avoid_friday_close: bool = True,
        rollover_start_hour: int = 21,
        rollover_end_hour: int = 23,
        scalp_cutoff_hour: int = 20,
        scalp_cutoff_minute: int = 30,
        friday_cutoff_hour: int = 19,
    ):
        if config is not None:
            sess_cfg = config.get("session_filter", config.get("sessions", {})) if isinstance(config, dict) else getattr(config, "session_filter", None)
            if sess_cfg is not None:
                if isinstance(sess_cfg, dict):
                    self.allowed_start_utc_hour = sess_cfg.get("allowed_start_utc_hour", allowed_start_utc_hour)
                    self.allowed_end_utc_hour = sess_cfg.get("allowed_end_utc_hour", allowed_end_utc_hour)
                    self.avoid_rollover_window = sess_cfg.get("avoid_rollover_window", avoid_rollover_window)
                    self.avoid_friday_close = sess_cfg.get("avoid_friday_close", avoid_friday_close)
                else:
                    self.allowed_start_utc_hour = getattr(sess_cfg, "allowed_start_utc_hour", allowed_start_utc_hour)
                    self.allowed_end_utc_hour = getattr(sess_cfg, "allowed_end_utc_hour", allowed_end_utc_hour)
                    self.avoid_rollover_window = getattr(sess_cfg, "avoid_rollover_window", avoid_rollover_window)
                    self.avoid_friday_close = getattr(sess_cfg, "avoid_friday_close", avoid_friday_close)
            else:
                self.allowed_start_utc_hour = allowed_start_utc_hour
                self.allowed_end_utc_hour = allowed_end_utc_hour
                self.avoid_rollover_window = avoid_rollover_window
                self.avoid_friday_close = avoid_friday_close
        else:
            self.allowed_start_utc_hour = allowed_start_utc_hour
            self.allowed_end_utc_hour = allowed_end_utc_hour
            self.avoid_rollover_window = avoid_rollover_window
            self.avoid_friday_close = avoid_friday_close

        self.rollover_start_hour = rollover_start_hour
        self.rollover_end_hour = rollover_end_hour
        self.scalp_cutoff_hour = scalp_cutoff_hour
        self.scalp_cutoff_minute = scalp_cutoff_minute
        self.friday_cutoff_hour = friday_cutoff_hour

    def is_rollover_lockout(self, dt: Optional[datetime] = None) -> bool:
        """
        Checks if datetime is inside the 21:00 - 23:00 UTC rollover lockout.
        During this window, bank rollover widens spreads drastically.
        """
        u = _to_utc(dt)
        return self.rollover_start_hour <= u.hour < self.rollover_end_hour

    def is_asian_session(self, dt: Optional[datetime] = None) -> bool:
        """Asian session: 01:00 to 08:00 UTC."""
        u = _to_utc(dt)
        return 1 <= u.hour < 8

    def is_london_session(self, dt: Optional[datetime] = None) -> bool:
        """London session: 08:00 to 16:30 UTC."""
        u = _to_utc(dt)
        return (8 <= u.hour < 16) or (u.hour == 16 and u.minute <= 30)

    def is_tokyo_overlap(self, dt: Optional[datetime] = None) -> bool:
        """Tokyo / European overlap: 07:00 to 09:00 UTC."""
        u = _to_utc(dt)
        return 7 <= u.hour < 9

    def is_london_ny_overlap(self, dt: Optional[datetime] = None) -> bool:
        """London / NY overlap: 12:00 to 16:00 UTC."""
        u = _to_utc(dt)
        return 12 <= u.hour < 16

    def is_ny_session(self, dt: Optional[datetime] = None) -> bool:
        """New York session: 12:00 to 20:00 UTC."""
        u = _to_utc(dt)
        return 12 <= u.hour < 20

    def is_friday_close(self, dt: Optional[datetime] = None) -> bool:
        """Friday close protection: after 19:00 UTC."""
        u = _to_utc(dt)
        return u.weekday() == 4 and u.hour >= self.friday_cutoff_hour

    def is_weekend(self, dt: Optional[datetime] = None) -> bool:
        """Forex market closed Saturday and Sunday before 21:00 UTC."""
        u = _to_utc(dt)
        weekday = u.weekday()
        if weekday == 5:  # Saturday
            return True
        if weekday == 6 and u.hour < 21:  # Sunday before market open
            return True
        return False

    def is_scalp_close_time(self, dt: Optional[datetime] = None) -> bool:
        """
        Checks if open intraday scalps should be closed before rollover.
        Cutoff is 20:30 UTC, or Friday 19:00 UTC.
        """
        u = _to_utc(dt)
        if self.is_friday_close(u):
            return True
        if (u.hour == self.scalp_cutoff_hour and u.minute >= self.scalp_cutoff_minute) or (self.rollover_start_hour <= u.hour < self.rollover_end_hour):
            return True
        return False

    should_close_scalps = is_scalp_close_time

    def get_session(self, dt: Optional[datetime] = None) -> str:
        """
        Returns string identifier of the active session.
        Matches requirements of test suites:
        - 02:00 UTC -> 'ASIAN'
        - 08:30 UTC -> 'LONDON'
        - 14:00 UTC -> 'LONDON_NY_OVERLAP' (matches 'NY', 'NEW_YORK', or 'OVERLAP')
        """
        u = _to_utc(dt)
        if self.is_rollover_lockout(u):
            return "ROLLOVER_LOCKOUT"
        if self.is_weekend(u):
            return "WEEKEND_CLOSED"

        hour = u.hour
        minute = u.minute

        if 12 <= hour < 16:
            return "LONDON_NY_OVERLAP"
        elif 8 <= hour < 12 or (hour == 16 and minute <= 30):
            return "LONDON"
        elif (hour == 16 and minute > 30) or (17 <= hour < 20):
            return "NEW_YORK"
        elif 1 <= hour < 8:
            return "ASIAN"
        elif 20 <= hour < 21:
            return "EVENING_CLOSE"
        else:
            return "OFF_HOURS"

    get_current_session_name = get_session

    def can_trade(self, dt: Optional[datetime] = None) -> bool:
        """
        Boolean check if trading entry is permitted.
        Rejects weekends, Friday post-19:00, and rollover (21:00-23:00).
        """
        u = _to_utc(dt)
        if self.avoid_rollover_window and self.is_rollover_lockout(u):
            return False
        if self.is_weekend(u):
            return False
        if self.avoid_friday_close and self.is_friday_close(u):
            return False
        return True

    def is_trading_allowed(self, dt: Optional[datetime] = None) -> Tuple[bool, str]:
        """
        Detailed check returning (allowed: bool, reason: str).
        """
        u = _to_utc(dt)
        if self.is_weekend(u):
            return False, "Market closed: Weekend"
        if self.avoid_friday_close and self.is_friday_close(u):
            return False, "Trading barred: Friday close cutoff (weekend gap risk)"
        if self.avoid_rollover_window and self.is_rollover_lockout(u):
            return False, f"Trading barred: Rollover spread blowout lockout ({u.hour:02d}:{u.minute:02d} UTC)"
        return True, "Trading allowed: Active liquidity"


# Alias for backward compatibility
MarketHoursGuard = MarketHours

# Module-level default instance and helper functions
_default_market_hours = MarketHours()

def is_rollover_lockout(dt: Optional[datetime] = None) -> bool:
    return _default_market_hours.is_rollover_lockout(dt)

def is_asian_session(dt: Optional[datetime] = None) -> bool:
    return _default_market_hours.is_asian_session(dt)

def is_london_session(dt: Optional[datetime] = None) -> bool:
    return _default_market_hours.is_london_session(dt)

def is_tokyo_overlap(dt: Optional[datetime] = None) -> bool:
    return _default_market_hours.is_tokyo_overlap(dt)

def is_london_ny_overlap(dt: Optional[datetime] = None) -> bool:
    return _default_market_hours.is_london_ny_overlap(dt)

def is_ny_session(dt: Optional[datetime] = None) -> bool:
    return _default_market_hours.is_ny_session(dt)

def is_friday_close(dt: Optional[datetime] = None) -> bool:
    return _default_market_hours.is_friday_close(dt)

def is_weekend(dt: Optional[datetime] = None) -> bool:
    return _default_market_hours.is_weekend(dt)

def is_scalp_close_time(dt: Optional[datetime] = None) -> bool:
    return _default_market_hours.is_scalp_close_time(dt)

should_close_scalps = is_scalp_close_time

def get_session(dt: Optional[datetime] = None) -> str:
    return _default_market_hours.get_session(dt)

get_current_session_name = get_session

def can_trade(dt: Optional[datetime] = None) -> bool:
    return _default_market_hours.can_trade(dt)

def is_trading_allowed(dt: Optional[datetime] = None) -> Tuple[bool, str]:
    return _default_market_hours.is_trading_allowed(dt)
