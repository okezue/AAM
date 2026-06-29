from __future__ import annotations
from aamemory.memory.sqlitestore import SQLiteEpisodeStore
from aamemory.schema import Episode, SparseCode
def testsqliteroundtripandpostings(tmp_path) -> None:
    store = SQLiteEpisodeStore(tmp_path / "memory.sqlite3")
    episode = Episode("e1", "hello", SparseCode.frommapping(32, {2: 1.0, 7: 0.5}))
    store.add(episode)
    loaded = store.get("e1")
    assert loaded is not None
    assert loaded.text == "hello"
    assert [item.episode_id for item in store.candidates([7])] == ["e1"]
    assert store.delete("e1")
    assert store.get("e1") is None
    store.close()
def testsqliteclearkeepsstoreusable(tmp_path) -> None:
    store = SQLiteEpisodeStore(tmp_path / "memory.sqlite3")
    store.add(Episode("e1", "one", SparseCode.frommapping(8, {1: 1.0})))
    store.clear()
    assert len(store) == 0
    store.add(Episode("e2", "two", SparseCode.frommapping(8, {2: 1.0})))
    assert store.get("e2") is not None
    store.close()
