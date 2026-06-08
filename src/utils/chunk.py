from dataclasses import dataclass

@dataclass
class Chunk:
    code: str
    header: str
    file: str
    start_line: int
    end_line: int
    language: str
    token_count: int
    score: float | None = None