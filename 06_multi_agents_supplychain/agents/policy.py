# agents/policy.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List
import yaml

@dataclass
class PolicyDecision:
    effect: str  # ALLOW | DENY | ESCALATE
    reason: str
    rule_id: str
    citations: List[str]

def _match_rule(when: Dict[str, Any], facts: Dict[str, Any]) -> bool:
    # Supported match keys:
    # - action: exact match
    # - role_in / role_not_in
    # - region_in
    for k, v in when.items():
        if k == "action":
            if facts.get("action") != v:
                return False
        elif k == "role_in":
            if facts.get("role") not in v:
                return False
        elif k == "role_not_in":
            if facts.get("role") in v:
                return False
        elif k == "region_in":
            if facts.get("region") not in v:
                return False
        else:
            # unknown condition type -> fail safe
            return False
    return True

class PolicyAgent:
    def __init__(self, policy_path: str = "config/policies.yaml"):
        with open(policy_path, "r", encoding="utf-8") as f:
            self.policy = yaml.safe_load(f)
        self.default_effect = self.policy.get("default_effect", "ESCALATE")
        self.rules = self.policy.get("rules", [])

    def decide(self, facts: Dict[str, Any]) -> PolicyDecision:
        for r in self.rules:
            when = r.get("when", {})
            if _match_rule(when, facts):
                return PolicyDecision(
                    effect=r.get("effect", self.default_effect),
                    reason=r.get("reason", "No reason provided."),
                    rule_id=r.get("id", "UNKNOWN"),
                    citations=r.get("citations", []),
                )
        return PolicyDecision(
            effect=self.default_effect,
            reason="No matching rule. Escalate to policy owner.",
            rule_id="DEFAULT",
            citations=[f"Policy bundle version {self.policy.get('version', 'unknown')}"],
        )