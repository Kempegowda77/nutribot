from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Deque, Dict, List, Optional

import torch


@dataclass
class SessionEntry:
    text: str
    value: float
    timestamp: datetime


class SessionMemory:
    """
    Store last N session entries with timestamp.

    Each entry contains:
    - text: raw user/system text
    - value: numeric signal (for lightweight modeling features)
    - timestamp: UTC time of insertion
    """

    def __init__(self, max_entries: int = 50) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be greater than 0.")
        self.max_entries = max_entries
        self._entries: Deque[SessionEntry] = deque(maxlen=max_entries)

    def add_entry(
        self,
        text: str,
        value: float,
        timestamp: Optional[datetime] = None,
    ) -> None:
        ts = timestamp or datetime.now(timezone.utc)
        self._entries.append(SessionEntry(text=text, value=float(value), timestamp=ts))

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)

    def to_list(self) -> List[Dict[str, object]]:
        return [
            {
                "text": entry.text,
                "value": entry.value,
                "timestamp": entry.timestamp.isoformat(),
            }
            for entry in self._entries
        ]

    def to_tensors(self) -> Dict[str, torch.Tensor]:
        """
        Returns tensorized view of memory:
        - values: [num_entries] float32
        - timestamps: [num_entries] int64 (unix seconds)
        """
        entries = list(self._entries)
        if not entries:
            return {
                "values": torch.empty(0, dtype=torch.float32),
                "timestamps": torch.empty(0, dtype=torch.int64),
            }

        values = torch.tensor([e.value for e in entries], dtype=torch.float32)
        timestamps = torch.tensor(
            [int(e.timestamp.timestamp()) for e in entries],
            dtype=torch.int64,
        )

        return {"values": values, "timestamps": timestamps}

