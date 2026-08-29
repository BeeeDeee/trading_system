from enum import Enum


class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"

    @property
    def sign(self) -> int:
        return 1 if self is Direction.LONG else -1


class Regime(str, Enum):
    """PER-SYMBOL derived label. See 05-FEATURES_AND_REGIME.md SS6."""

    TREND_UP = "TREND_UP"
    TREND_DOWN = "TREND_DOWN"
    RANGE = "RANGE"
    CHOP = "CHOP"
    UNKNOWN = "UNKNOWN"  # warm-up incomplete; blocks all strategies


class MarketRegime(str, Enum):
    """MARKET-WIDE label, computed once per timestamp from the benchmark index
    and broadcast to every symbol's row. See 05-FEATURES_AND_REGIME.md SS7.

    Distinct from `Regime`, and the distinction is load-bearing: `Regime` is one
    symbol's own trend structure, `MarketRegime` is the environment. A stock can
    be TREND_UP in a RISK_OFF market, and that combination is precisely what the
    market gate is there to refuse.
    """

    RISK_ON = "RISK_ON"
    NEUTRAL = "NEUTRAL"
    RISK_OFF = "RISK_OFF"
    UNKNOWN = "UNKNOWN"  # benchmark warm-up incomplete; blocks everything


class VolBucket(str, Enum):
    """ATR percentile vs the symbol's own trailing 2-year distribution."""

    LOW = "LOW"  # percentile < 0.33
    MID = "MID"  # 0.33 <= percentile < 0.67
    HIGH = "HIGH"  # percentile >= 0.67
    UNKNOWN = "UNKNOWN"


class SetupOutcome(str, Enum):
    TARGET = "TARGET"  # target touched first (or gapped through, in our favour)
    STOP = "STOP"  # stop touched first, gapped through, or both in one bar
    TIME = "TIME"  # max_hold_bars reached; exit at that session's close
    OPEN = "OPEN"  # not yet resolved; excluded from edge statistics


class RejectionReason(str, Enum):
    """Closed set. Every rejected candidate carries exactly one of these.
    Never write a free-text reason: the decision log is queried by this column.
    """

    NOT_IN_UNIVERSE = "NOT_IN_UNIVERSE"
    INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"  # warm-up bars missing
    DATA_GAP = "DATA_GAP"
    DATA_SUSPECT = "DATA_SUSPECT"  # failed a quality check
    STALE_DATA = "STALE_DATA"
    LOW_PRICE = "LOW_PRICE"  # close_raw below min_price_usd
    LOW_LIQUIDITY = "LOW_LIQUIDITY"
    WIDE_SPREAD = "WIDE_SPREAD"
    MARKET_REGIME_BLOCKED = "MARKET_REGIME_BLOCKED"
    REGIME_BLOCKED = "REGIME_BLOCKED"  # per-symbol regime
    EARNINGS_IN_WINDOW = "EARNINGS_IN_WINDOW"
    HARD_TO_BORROW = "HARD_TO_BORROW"  # shorts only
    THIN_CROSS_SECTION = "THIN_CROSS_SECTION"  # xs_population too small
    NO_SETUP = "NO_SETUP"
    INSUFFICIENT_BIN_SAMPLES = "INSUFFICIENT_BIN_SAMPLES"
    BELOW_EV_THRESHOLD = "BELOW_EV_THRESHOLD"
    COST_UNAVAILABLE = "COST_UNAVAILABLE"
    ALREADY_IN_POSITION = "ALREADY_IN_POSITION"
    PORTFOLIO_HEAT_CAP = "PORTFOLIO_HEAT_CAP"
    CLUSTER_CAP = "CLUSTER_CAP"
    BETA_CAP = "BETA_CAP"
    GROSS_EXPOSURE_CAP = "GROSS_EXPOSURE_CAP"
    MAX_POSITIONS = "MAX_POSITIONS"
    BELOW_TOP_N = "BELOW_TOP_N"
    SENTIMENT_VETO = "SENTIMENT_VETO"
    SIZE_BELOW_MIN_NOTIONAL = "SIZE_BELOW_MIN_NOTIONAL"
    KILL_SWITCH = "KILL_SWITCH"
    CIRCUIT_BREAKER = "CIRCUIT_BREAKER"


class ActionType(str, Enum):
    SPLIT = "SPLIT"
    DIVIDEND = "DIVIDEND"
    SPINOFF = "SPINOFF"
    MERGER = "MERGER"
    TICKER_CHANGE = "TICKER_CHANGE"


class OrderType(str, Enum):
    MARKET = "MARKET"  # market-on-open for entries
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET"
    TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"


class RunMode(str, Enum):
    BACKTEST = "BACKTEST"
    PAPER = "PAPER"
    LIVE = "LIVE"
