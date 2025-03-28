"""A multi-GPU-friendly python command line logger."""

import logging
from collections.abc import Mapping

from lightning_utilities.core.rank_zero import (
    rank_prefixed_message,
    rank_zero_only,
)


class RankedLogger(logging.LoggerAdapter):
    r"""Initialize a multi-GPU-friendly python command line logger.

    The logger logs on all processes with their rank prefixed in the log message.

    Parameters
    ----------
    name : str, optional
        The name of the logger, by default __name__.
    rank_zero_only : bool, optional
        Whether to force all logs to only occur on the rank zero process (default: False).
    extra : Mapping[str, object], optional
        A dict-like object which provides contextual information. See `logging.LoggerAdapter` for more information (default: None).
    """

    def __init__(
        self,
        name: str = __name__,
        rank_zero_only: bool = False,
        extra: Mapping[str, object] | None = None,
    ) -> None:
        logger = logging.getLogger(name)
        super().__init__(logger=logger, extra=extra)
        self.rank_zero_only = rank_zero_only

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.logger.name!r}, rank_zero_only={self.rank_zero_only!r}, extra={self.extra})"

    def log(
        self, level: int, msg: str, rank: int | None = None, *args, **kwargs
    ) -> None:
        r"""Delegate a log call to the underlying logger.

        The function first prefixes the message with the rank of the process it's being logged from and then logs the message. If `'rank'` is provided, then the log will only occur on that rank/process.

        Parameters
        ----------
        level : int
            The level to log at. Look at `logging.__init__.py` for more information.
        msg : str
            The message to log.
        rank : int, optional
            The rank to log at (default: None).
        *args : Any
            Additional args to pass to the underlying logging function.
        **kwargs : Any
            Any additional keyword args to pass to the underlying logging function.
        """
        if self.isEnabledFor(level):
            msg, kwargs = self.process(msg, kwargs)
            current_rank = getattr(rank_zero_only, "rank", None)
            if current_rank is None:
                raise RuntimeError(
                    "The `rank_zero_only.rank` needs to be set before use"
                )
            msg = rank_prefixed_message(msg, current_rank)
            if self.rank_zero_only:
                if current_rank == 0:
                    self.logger.log(level, msg, *args, **kwargs)
            else:
                if rank is None or current_rank == rank:
                    self.logger.log(level, msg, *args, **kwargs)
