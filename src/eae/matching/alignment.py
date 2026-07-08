# src/eae/matching/alignment.py

from __future__ import annotations

from typing import List


def alignment_cost(
    pattern: List[str],
    segment: List[str],
) -> int:
    """
    Alignment/edit cost between a dictionary path and a case segment.

    Cost model:
      synchronous move = 0
      log move         = 1
      model move       = 1
      mismatch         = 2 via one log move + one model move
    """
    m = len(pattern)
    n = len(segment)

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i

    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pattern[i - 1] == segment[j - 1]:
                sync = dp[i - 1][j - 1]
            else:
                sync = dp[i - 1][j - 1] + 2

            model_move = dp[i - 1][j] + 1
            log_move = dp[i][j - 1] + 1

            dp[i][j] = min(sync, model_move, log_move)

    return int(dp[m][n])