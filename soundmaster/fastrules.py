"""Быстрые правила на Mac: слова-подводки взводят синхронный эффект, слова-триггеры его запускают.

Слова берутся из реестра (arm_keywords / fire_keywords). Отрицания и отмены («не», «хотя нет»,
«не надо») блокируют взведение: «сейчас вылетит птичка… хотя нет» — не намерение.
Текст приходит после конца фразы, поэтому задержка ≈ пауза VAD + распознавание; потоковое
распознавание слов — следующий шаг, если этого окажется мало.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .registry import Registry

CANCEL_PHRASES = ("хотя нет", "не надо", "отмена", "передумал", "передумала", "не буду", "не будем")
NEGATIONS = ("не", "нет", "ни", "без")


def normalize(text: str) -> list[str]:
    t = text.lower().replace("ё", "е")
    return re.findall(r"[a-zа-я0-9-]+", t)


def _norm_kw(k: str) -> list[str]:
    return normalize(k)


@dataclass
class RuleHit:
    kind: str  # "arm" | "fire" | "cancel"
    effect: str | None
    word: str
    negated: bool = False


class FastRules:
    def __init__(self, registry: Registry, negation_window: int = 2):
        self.registry = registry
        self.window = negation_window
        self.arm_words: list[tuple[list[str], str, str]] = []  # (токены, эффект, исходное слово)
        self.fire_words: list[tuple[list[str], str, str]] = []
        for e in registry.effects.values():
            if e.human_only:
                continue  # громкие эффекты не взводятся и не запускаются словами
            for k in e.arm_keywords:
                self.arm_words.append((_norm_kw(k), e.id, k))
            for k in e.fire_keywords:
                self.fire_words.append((_norm_kw(k), e.id, k))

    @staticmethod
    def _find(tokens: list[str], kw: list[str]) -> list[int]:
        n = len(kw)
        # Совпадение по началу слова: «птичку», «птичка» → «птичк…».
        stem = [k[:-1] if len(k) > 4 else k for k in kw]
        return [i for i in range(len(tokens) - n + 1)
                if all(tokens[i + j].startswith(stem[j]) for j in range(n))]

    def _negated(self, tokens: list[str], i: int) -> bool:
        return any(t in NEGATIONS for t in tokens[max(0, i - self.window): i])

    def analyze(self, text: str) -> list[RuleHit]:
        tokens = normalize(text)
        low = " ".join(tokens)
        hits: list[RuleHit] = []
        cancelled = any(c in low for c in CANCEL_PHRASES)
        for kw, eff, word in self.arm_words:
            for i in self._find(tokens, kw):
                hits.append(RuleHit("arm", eff, word, negated=self._negated(tokens, i) or cancelled))
        for kw, eff, word in self.fire_words:
            for i in self._find(tokens, kw):
                hits.append(RuleHit("fire", eff, word, negated=self._negated(tokens, i)))
        if cancelled:
            hits.append(RuleHit("cancel", None, "отмена"))
        return hits
