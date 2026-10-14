"""Неизменяемые записи источника, кандидата и результата проверки."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceIdentity:
    document_id: str
    content_sha256: str
    revision_id: str | None = None


@dataclass(frozen=True)
class Source:
    identity: SourceIdentity
    text: str


@dataclass(frozen=True)
class SupportingQuotation:
    text: str
    start: int | None
    end: int | None


@dataclass(frozen=True)
class ClaimCandidate:
    candidate_id: str
    source: SourceIdentity
    statement: str
    quotation: SupportingQuotation
    conditions: tuple[str, ...]
    exceptions: tuple[str, ...]
    scope: str | None
    jurisdiction: str | None
    effective_date: str | None


@dataclass(frozen=True)
class ValidationResult:
    candidate: ClaimCandidate
    errors: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """Пройдены ли проверки структуры и происхождения; юридическая проверка не выполняется."""
        return not self.errors
