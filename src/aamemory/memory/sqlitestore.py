from __future__ import annotations
import json
import sqlite3
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any
from aamemory.memory.store import EpisodeStore
from aamemory.schema import Episode, SourceRef, SparseCode
def sourcetodict(source: SourceRef) -> dict[str, Any]:
    return {
        "uri": source.uri,
        "document_id": source.document_id,
        "start": source.start,
        "end": source.end,
        "checksum": source.checksum,
        "metadata": dict(source.metadata),
    }
def episodetojson(episode: Episode) -> str:
    return json.dumps(
        {
            "episode_id": episode.episode_id,
            "text": episode.text,
            "code": episode.code.tojsonable(),
            "timestamp": episode.timestamp,
            "salience": episode.salience,
            "confidence": episode.confidence,
            "source": sourcetodict(episode.source),
            "payload": dict(episode.payload),
            "metadata": episode.metadata,
        },
        separators=(",", ":"),
        ensure_ascii=False,
    )
def episodefromjson(value: str) -> Episode:
    raw = json.loads(value)
    source_raw = raw.get("source", {})
    return Episode(
        episode_id=raw["episode_id"],
        text=raw["text"],
        code=SparseCode.fromjsonable(raw["code"]),
        timestamp=raw["timestamp"],
        salience=float(raw.get("salience", 1.0)),
        confidence=float(raw.get("confidence", 1.0)),
        source=SourceRef(
            uri=source_raw.get("uri"),
            document_id=source_raw.get("document_id"),
            start=source_raw.get("start"),
            end=source_raw.get("end"),
            checksum=source_raw.get("checksum"),
            metadata=source_raw.get("metadata", {}),
        ),
        payload=raw.get("payload", {}),
        metadata=dict(raw.get("metadata", {})),
    )
class SQLiteEpisodeStore(EpisodeStore):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.path))
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS episodes (
                episode_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS postings (
                feature INTEGER NOT NULL,
                episode_id TEXT NOT NULL,
                PRIMARY KEY (feature, episode_id),
                FOREIGN KEY (episode_id) REFERENCES episodes(episode_id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_postings_feature ON postings(feature);
            """
        )
        self.connection.commit()
    def add(self, episode: Episode) -> None:
        with self.connection:
            self.connection.execute("DELETE FROM postings WHERE episode_id = ?", (episode.episode_id,))
            self.connection.execute(
                "INSERT OR REPLACE INTO episodes(episode_id, timestamp, json) VALUES (?, ?, ?)",
                (episode.episode_id, episode.timestamp, episodetojson(episode)),
            )
            self.connection.executemany(
                "INSERT OR IGNORE INTO postings(feature, episode_id) VALUES (?, ?)",
                ((int(feature), episode.episode_id) for feature in episode.code.indices),
            )
    def get(self, episode_id: str) -> Episode | None:
        row = self.connection.execute(
            "SELECT json FROM episodes WHERE episode_id = ?", (episode_id,)
        ).fetchone()
        return episodefromjson(row[0]) if row else None
    def delete(self, episode_id: str) -> bool:
        with self.connection:
            self.connection.execute("DELETE FROM postings WHERE episode_id = ?", (episode_id,))
            cursor = self.connection.execute(
                "DELETE FROM episodes WHERE episode_id = ?", (episode_id,)
            )
        return cursor.rowcount > 0
    def candidates(self, feature_indices: Iterable[int], limit: int = 500) -> list[Episode]:
        features = sorted(set(int(x) for x in feature_indices))
        if not features:
            return []
        votes: dict[str, int] = defaultdict(int)
        for start in range(0, len(features), 500):
            chunk = features[start : start + 500]
            placeholders = ",".join("?" for _ in chunk)
            query = f"SELECT episode_id, COUNT(*) FROM postings WHERE feature IN ({placeholders}) GROUP BY episode_id"
            for episode_id, count in self.connection.execute(query, chunk):
                votes[str(episode_id)] += int(count)
        ids = sorted(votes, key=lambda eid: (-votes[eid], eid))[:limit]
        return [episode for eid in ids if (episode := self.get(eid)) is not None]
    def all(self) -> Iterator[Episode]:
        cursor = self.connection.execute("SELECT json FROM episodes ORDER BY timestamp, episode_id")
        for (value,) in cursor:
            yield episodefromjson(value)
    def __len__(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM episodes").fetchone()[0])
    def clear(self) -> None:
        with self.connection:
            self.connection.execute("DELETE FROM postings")
            self.connection.execute("DELETE FROM episodes")
    def stats(self) -> dict[str, Any]:
        posting_features, posting_entries = self.connection.execute(
            "SELECT COUNT(DISTINCT feature), COUNT(*) FROM postings"
        ).fetchone()
        physical_bytes = 0
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(str(self.path) + suffix)
            if candidate.exists():
                physical_bytes += candidate.stat().st_size
        return {
            "episodes": len(self),
            "posting_features": int(posting_features),
            "posting_entries": int(posting_entries),
            "physical_bytes": int(physical_bytes),
        }
    def close(self) -> None:
        self.connection.close()
