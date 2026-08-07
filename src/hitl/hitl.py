"""
Lab 11 — Part 4: Human-in-the-Loop Design
  TODO 11: Confidence Router
  TODO 12: Design 3 HITL decision points
"""
from dataclasses import dataclass


# ============================================================
# TODO 11: Implement ConfidenceRouter
#
# Route agent responses based on confidence scores:
#   - HIGH (>= 0.9): Auto-send to user
#   - MEDIUM (0.7 - 0.9): Queue for human review
#   - LOW (< 0.7): Escalate to human immediately
#
# Special case: if the action is HIGH_RISK (e.g., money transfer,
# account deletion), ALWAYS escalate regardless of confidence.
#
# Implement the route() method.
# ============================================================

HIGH_RISK_ACTIONS = [
    "transfer_money",
    "close_account",
    "change_password",
    "delete_data",
    "update_personal_info",
]


@dataclass
class RoutingDecision:
    """Result of the confidence router."""
    action: str          # "auto_send", "queue_review", "escalate"
    confidence: float
    reason: str
    priority: str        # "low", "normal", "high"
    requires_human: bool


class ConfidenceRouter:
    """Route agent responses based on confidence and risk level.

    Thresholds:
        HIGH:   confidence >= 0.9 -> auto-send
        MEDIUM: 0.7 <= confidence < 0.9 -> queue for review
        LOW:    confidence < 0.7 -> escalate to human

    High-risk actions always escalate regardless of confidence.
    """

    HIGH_THRESHOLD = 0.9
    MEDIUM_THRESHOLD = 0.7

    def route(self, response: str, confidence: float,
              action_type: str = "general") -> RoutingDecision:
        """Route a response based on confidence score and action type.

        Args:
            response: The agent's response text
            confidence: Confidence score between 0.0 and 1.0
            action_type: Type of action (e.g., "general", "transfer_money")

        Returns:
            RoutingDecision with routing action and metadata
        """
        if action_type in HIGH_RISK_ACTIONS:
            return RoutingDecision(
                action="escalate",
                confidence=confidence,
                reason=f"High-risk action: {action_type}",
                priority="high",
                requires_human=True,
            )

        if confidence >= self.HIGH_THRESHOLD:
            return RoutingDecision(
                action="auto_send",
                confidence=confidence,
                reason="High confidence",
                priority="low",
                requires_human=False,
            )

        if confidence >= self.MEDIUM_THRESHOLD:
            return RoutingDecision(
                action="queue_review",
                confidence=confidence,
                reason="Medium confidence — needs review",
                priority="normal",
                requires_human=True,
            )

        return RoutingDecision(
            action="escalate",
            confidence=confidence,
            reason="Low confidence — escalating",
            priority="high",
            requires_human=True,
        )


# ============================================================
# TODO 12: Design 3 HITL decision points + a review lifecycle
#
# For each decision point, define:
# - trigger: What condition activates this HITL check?
# - hitl_model: Which model? (human-in-the-loop, human-on-the-loop,
#   human-as-tiebreaker)
# - context_needed: What info does the human reviewer need?
# - example: A concrete scenario
# - approval_path: What approve/reject/timeout decision is recorded?
# - audit_fields: Which correlation ID, intent and proposed action/diff are logged?
#
# Think about real banking scenarios where human judgment is critical.
# ============================================================

hitl_decision_points = [
    {
        "id": 1,
        "name": "Beneficiary change on a money transfer",
        "trigger": (
            "action_type == 'transfer_money' AND the request adds/changes the "
            "recipient account (new or edited beneficiary), regardless of the "
            "model's confidence score — HIGH_RISK_ACTIONS always route away "
            "from auto_send."
        ),
        "hitl_model": "human-in-the-loop",
        "context_needed": (
            "Old beneficiary (name + account number) vs new beneficiary, "
            "transfer amount, customer's stated reason, account tenure, and "
            "any anomaly signals (first transfer to this beneficiary, amount "
            "far above the customer's usual pattern, beneficiary added minutes "
            "before the transfer)."
        ),
        "example": (
            "Customer chat asks to send 500,000,000 VND to a beneficiary added "
            "to the profile 10 minutes earlier. Confidence=0.97 but action_type="
            "'transfer_money' forces route()->'escalate'; a reviewer must see the "
            "old/new beneficiary diff and the anomaly flag before anything is sent."
        ),
        "approval_path": (
            "approve -> queued transfer is released to the payment gateway and "
            "is_egress_allowed() still re-checked before the call; reject -> "
            "transfer is cancelled and customer is notified; timeout (no reviewer "
            "decision within SLA, e.g. 15 min) -> request is held in 'pending' "
            "state and the transfer is NOT sent — a fail-closed default, never "
            "an implicit approve."
        ),
        "audit_fields": (
            "request_id (correlation ID), customer_id, action_type, intent, "
            "old_beneficiary/new_beneficiary diff, amount, confidence, "
            "reviewer_id, reviewer_decision (approve/reject/timeout), "
            "decision_timestamp, sla_deadline — written to audit_log before "
            "and after the reviewer decision."
        ),
    },
    {
        "id": 2,
        "name": "Irreversible account lifecycle change",
        "trigger": (
            "action_type in {'close_account', 'delete_data'} — any request that "
            "cannot be cleanly undone once executed."
        ),
        "hitl_model": "human-in-the-loop",
        "context_needed": (
            "Account balance and open products (loans/cards) that must be "
            "settled first, customer identity verification status, stated "
            "reason for closure/deletion, and whether the request originated "
            "from an authenticated session or an external email/RAG document "
            "(untrusted content must never trigger this on its own)."
        ),
        "example": (
            "An email summarized by the agent contains 'please close this "
            "account'. Because the instruction came from untrusted external "
            "content (Checkpoint 1) it is never auto-executed; if the customer "
            "also confirms in-session, the request still routes to escalate "
            "and a reviewer must approve the closure before it happens."
        ),
        "approval_path": (
            "approve -> reviewer countersigns and the closure/deletion job runs "
            "with the reviewer_id attached; reject -> request closed, customer "
            "informed with a reason; timeout -> request auto-rejects (safer "
            "default for irreversible actions than an indefinite hold), and the "
            "customer must resubmit."
        ),
        "audit_fields": (
            "request_id, customer_id, action_type, intent, proposed diff (account "
            "status before/after), source_of_request (session vs external "
            "document), reviewer_id, reviewer_decision, decision_timestamp."
        ),
    },
    {
        "id": 3,
        "name": "Low/medium-confidence advice on regulated topics",
        "trigger": (
            "action_type == 'general' (no side effect) but ConfidenceRouter "
            "returns 'queue_review' (confidence 0.70–<0.90) or 'escalate' "
            "(confidence < 0.70) — e.g. interest-rate or loan-eligibility "
            "answers the model is not confident about."
        ),
        "hitl_model": "human-as-tiebreaker",
        "context_needed": (
            "The model's draft answer, confidence score, the RAG passages it "
            "cited, and the ground-truth policy/rate sheet so the reviewer can "
            "confirm or correct the figure before it reaches the customer."
        ),
        "example": (
            "Customer asks for the 12-month savings rate; the model drafts "
            "5.5% with confidence 0.8 while the policy sheet says 4.25%. "
            "queue_review holds the draft so a reviewer corrects the number "
            "instead of a hallucinated rate reaching the customer."
        ),
        "approval_path": (
            "approve -> draft is sent to the customer as-is; reject/edit -> "
            "reviewer supplies the corrected answer, which is sent instead and "
            "stored as the source of truth for that request; timeout -> reply "
            "is held and the customer receives a 'still checking, will follow "
            "up' message rather than an unverified auto-send."
        ),
        "audit_fields": (
            "request_id, customer_id, intent, model_draft_answer, confidence, "
            "cited_sources, reviewer_id, reviewer_decision, final_answer_sent, "
            "decision_timestamp."
        ),
    },
]


# ============================================================
# Quick tests
# ============================================================

def test_confidence_router():
    """Test ConfidenceRouter with sample scenarios."""
    router = ConfidenceRouter()

    test_cases = [
        ("Balance inquiry", 0.95, "general"),
        ("Interest rate question", 0.82, "general"),
        ("Ambiguous request", 0.55, "general"),
        ("Transfer $50,000", 0.98, "transfer_money"),
        ("Close my account", 0.91, "close_account"),
    ]

    print("Testing ConfidenceRouter:")
    print("=" * 80)
    print(f"{'Scenario':<25} {'Conf':<6} {'Action Type':<18} {'Decision':<15} {'Priority':<10} {'Human?'}")
    print("-" * 80)

    for scenario, conf, action_type in test_cases:
        decision = router.route(scenario, conf, action_type)
        print(
            f"{scenario:<25} {conf:<6.2f} {action_type:<18} "
            f"{decision.action:<15} {decision.priority:<10} "
            f"{'Yes' if decision.requires_human else 'No'}"
        )

    print("=" * 80)


def test_hitl_points():
    """Display HITL decision points."""
    print("\nHITL Decision Points:")
    print("=" * 60)
    for point in hitl_decision_points:
        print(f"\n  Decision Point #{point['id']}: {point['name']}")
        print(f"    Trigger:  {point['trigger']}")
        print(f"    Model:    {point['hitl_model']}")
        print(f"    Context:  {point['context_needed']}")
        print(f"    Example:  {point['example']}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_confidence_router()
    test_hitl_points()
