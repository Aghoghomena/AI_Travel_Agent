import dspy
from src.utils.config import llm

# ── 1. Configure the LM ──────────────────────────────────────
lm = dspy.LM("openai/gpt-4o-mini")  # or whatever model you use
dspy.configure(lm=lm)

# ── 2. Define the signature (replaces your prompt template) ──
class IntentClassifier(dspy.Signature):
    """Classify a travel query as READY, NEEDS_INFO, or OUT_OF_SCOPE."""
    
    user_message: str = dspy.InputField(desc="The user's message")
    query_state: str  = dspy.InputField(desc="Current known travel details as JSON")
    
    status: str       = dspy.OutputField(desc="One of: READY, NEEDS_INFO, OUT_OF_SCOPE")
    detected_fields: str = dspy.OutputField(desc="Extracted travel details as JSON")
    reason: str       = dspy.OutputField(desc="Reason if OUT_OF_SCOPE, else empty")


class Elicitation(dspy.Signature):
    """Generate a question to collect missing travel information."""
    
    user_message: str      = dspy.InputField(desc="The user's message")
    query_state: str       = dspy.InputField(desc="Current known travel details as JSON")
    conversation_history: str = dspy.InputField(desc="Prior conversation as JSON")
    
    question: str          = dspy.OutputField(desc="Question to ask the user")
    updated_fields: str    = dspy.OutputField(desc="Any newly extracted fields as JSON")


# ── 3. Define the modules (replaces your node functions) ─────
class IntentClassifierModule(dspy.Module):
    def __init__(self):
        self.classify = dspy.ChainOfThought(IntentClassifier)
    
    def forward(self, user_message: str, query_state: dict) -> dict:
        result = self.classify(
            user_message=user_message,
            query_state=str(query_state)
        )
        return {
            "status": result.status,
            "detected_fields": result.detected_fields,
            "reason": result.reason,
        }


class ElicitationModule(dspy.Module):
    def __init__(self):
        self.elicit = dspy.ChainOfThought(Elicitation)
    
    def forward(self, user_message: str, query_state: dict, history: list) -> dict:
        result = self.elicit(
            user_message=user_message,
            query_state=str(query_state),
            conversation_history=str(history)
        )
        return {
            "question": result.question,
            "updated_fields": result.updated_fields,
        }


# ── 4. Training examples for optimisation ────────────────────
trainset = [
    dspy.Example(
        user_message="I want to travel",
        query_state="{}",
        status="NEEDS_INFO",
        detected_fields="{}",
        reason=""
    ).with_inputs("user_message", "query_state"),

    dspy.Example(
        user_message="Me and John, Lagos and London, July, 7 nights, anywhere in Europe",
        query_state="{}",
        status="READY",
        detected_fields='{"travellers": [{"name": "Me", "origin_city": "Lagos"}, {"name": "John", "origin_city": "London"}], "travel_month": "July", "duration_nights": 7}',
        reason=""
    ).with_inputs("user_message", "query_state"),

    dspy.Example(
        user_message="What is the weather in Lagos?",
        query_state="{}",
        status="OUT_OF_SCOPE",
        detected_fields="{}",
        reason="weather query"
    ).with_inputs("user_message", "query_state"),
]


# ── 5. Metric ─────────────────────────────────────────────────
def intent_metric(example, prediction, trace=None) -> bool:
    return example.status.upper() == prediction.status.upper().strip()


# ── 6. Optimise ───────────────────────────────────────────────
def optimise():
    classifier = IntentClassifierModule()
    
    optimizer = dspy.MIPROv2(metric=intent_metric, auto="light")
    optimised = optimizer.compile(classifier, trainset=trainset)
    
    # Save optimised prompts
    optimised.save("optimised_intent_classifier.json")
    return optimised


# ── 7. Use in your existing flow ──────────────────────────────
classifier = IntentClassifierModule()
# classifier.load("optimised_intent_classifier.json")  # load optimised version

def build_intent_classifier_prompt_dspy(user_message: str, query_state: dict) -> dict:
    return classifier(user_message=user_message, query_state=query_state)