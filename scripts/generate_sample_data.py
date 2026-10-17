"""Generate a realistic mock dataset matching the Name That LLM competition schema.

Useful for:
- Testing and debugging feature extractors, cross-validation, and submission pipelines
- Running unit tests without needing external downloads
- Offline rapid development
"""

import random
from pathlib import Path
import pandas as pd
from rich.console import Console

from name_that_llm.config import SAMPLE_DATA_DIR, TARGET_MODELS

console = Console()

# Sample prompt themes to simulate unseen prompt grouping
PROMPT_THEMES = [
    ("p001", "Explain the concept of quantum entanglement and its implications for information theory."),
    ("p002", "Write an analysis of the themes of fate and free will in Shakespeare's Macbeth."),
    ("p003", "Describe how modern compiler optimizations implement dead code elimination and loop unrolling."),
    ("p004", "Discuss the economic trade-offs between fixed exchange rate regimes and floating currency systems."),
    ("p005", "How do neural crest cells contribute to vertebrate development and craniofacial morphology?"),
    ("p006", "Explain the Riemann Hypothesis in accessible terms and why prime number distribution matters."),
    ("p007", "What are the ethical implications of autonomous weapons systems under international humanitarian law?"),
    ("p008", "Provide a creative prose narrative depicting a solitary lighthouse keeper during an oceanic anomaly."),
    ("p009", "Contrast existentialism with absurdism, focusing on Sartre and Camus."),
    ("p010", "Explain the mechanism of CRISPR-Cas9 base editing compared to prime editing."),
    # Prompts for test set (unseen in train)
    ("p101_test", "Discuss the architectural differences between Transformer attention and State Space Models (SSMs)."),
    ("p102_test", "Write a philosophical meditation on whether consciousness can be instantiated on silicon hardware."),
]

# Stylistic templates to mimic frontier LLM text signatures
MODEL_STYLE_FRAGMENTS = {
    # OpenAI styles (often structured with clear transitions, bullet points, concise summary)
    "gpt-5.5": "In summary, {prompt_topic} reveals key insights: 1. Core principles dictate operational outcomes. 2. Moreover, empirical observations corroborate this hypothesis. Crucially, the system remains robust under variable conditions.",
    "gpt-5.6-luna": "To understand {prompt_topic}, we must evaluate the underlying mechanisms. Furthermore, modern frameworks emphasize adaptability. Let us highlight the primary dimensions:\n- Theoretical basis\n- Practical deployment\nUltimately, this is foundational.",
    "gpt-5.6-sol": "Essentially, {prompt_topic} represents a dual phenomenon. On one hand, direct effects are immediately noticeable; on the other hand, downstream implications unfold progressively over time. Hence, careful parameterization is necessary.",
    "gpt-5.6-terra": "When analyzing {prompt_topic}, three elements emerge as decisive. First, structural integrity must be preserved. Second, interface constraints require continuous calibration. Finally, system resilience guarantees sustained performance.",
    "gpt-6-astra": "A rigorous examination of {prompt_topic} necessitates decoupling signal from noise. In particular, mathematical models predict asymptotic convergence under standard assumptions. Therefore, we conclude that stability is guaranteed.",
    
    # Google styles (often direct, factual, clear scientific terminology)
    "gemini-3.1-pro-preview": "At its core, {prompt_topic} functions through well-defined interactions. The governing equations demonstrate that state transitions follow predictable trajectories. Notably, recent advances in experimental verification confirm these predictions.",
    "gemini-3.6-flash": "Here is a breakdown of {prompt_topic}. Specifically, two complementary pathways exist: first, baseline pathways provide initial stability; second, dynamic responses adapt to disturbances. This ensures overall equilibrium.",
    "gemini-3.7-flash": "Understanding {prompt_topic} requires examining both micro-level dynamics and macro-level outcomes. Key factors include rate constants, boundary limits, and environmental feedback loops. Consequently, performance scales monotonically.",
    "gemini-3.8-flash": "Let's examine {prompt_topic}. In physical terms, the system minimizes potential energy while maximizing entropy production. As a result, the observed behavior naturally aligns with thermodynamic expectations.",
    
    # Anthropic styles (often nuanced, epistemic hedging, semicolons, em-dashes)
    "claude-fable-5": "When we consider {prompt_topic}, the picture is delightfully nuanced. It is not merely a technical problem—though technical rigor is indispensable—but rather an inquiry into how disparate elements weave into a coherent tapestry.",
    "claude-opus-4-6": "A thoughtful exploration of {prompt_topic} suggests that intuition can sometimes mislead; what appears straightforward on the surface often conceals significant subtleties beneath. Perhaps most intriguing is how these dynamics interlock.",
    "claude-opus-4-8": "It is worth distinguishing between the primary mechanisms of {prompt_topic} and their secondary manifestations. While the former dictate immediate outcomes, the latter shape long-term evolutionary trajectories—a crucial distinction for theorists.",
    "claude-opus-5": "To approach {prompt_topic} with appropriate care, one must acknowledge the inherent trade-offs involved. There are seldom singular answers; instead, one finds competing priorities balanced against specific contextual constraints.",
    "claude-sonnet-5": "Examining {prompt_topic} reveals an elegant interplay of principles. Specifically, three interrelated factors warrant attention: first, fundamental constraints; second, adaptive mechanisms; and third, systemic feedback.",
    
    # Qwen styles (often dense, structured, academic cadence)
    "qwen3.8-max": "Regarding {prompt_topic}, the standard formulation relies on rigorous optimization principles. By decomposing the problem into discrete modules, we observe that computational complexity scales quadratically under naive assumptions.",
    
    # ZAI (GLM) styles
    "glm-5.3": "Analysis of {prompt_topic} demonstrates that multi-tiered architectures provide superior representational capacity. In practical applications, parameter tuning must balance expressiveness with generalization error.",
    "glm-5.3-flash": "Briefly stated, {prompt_topic} can be characterized by its operational efficiency and structural modularity. Experimental data indicates that throughput increases proportionally with resource allocation.",
    
    # Moonshot AI (Kimi) styles
    "kimi-k3": "When dissecting {prompt_topic}, comprehensive context is vital. Through thorough investigation across historical and contemporary benchmarks, clear behavioral regularities become apparent, guiding optimal decision-making.",
}


def generate_sample_data(num_train_prompts: int = 8, num_test_prompts: int = 2) -> None:
    """Generate realistic synthetic train.csv and test.csv in data/sample."""
    SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(42)

    train_rows = []
    test_rows = []

    # 1. Generate Training Data
    train_prompts = PROMPT_THEMES[:num_train_prompts]
    row_id = 1
    for prompt_id, prompt_text in train_prompts:
        topic_short = prompt_text.split()[1:4]
        topic_str = " ".join(topic_short)
        for model in TARGET_MODELS:
            template = MODEL_STYLE_FRAGMENTS[model]
            body = template.format(prompt_topic=topic_str)
            train_rows.append(
                {
                    "id": f"t_{row_id:04d}",
                    "prompt_id": prompt_id,
                    "text": body,
                    "model": model,
                }
            )
            row_id += 1

    df_train = pd.DataFrame(train_rows)
    train_path = SAMPLE_DATA_DIR / "train.csv"
    df_train.to_csv(train_path, index=False)

    # 2. Generate Test Data (Unseen prompts)
    test_prompts = PROMPT_THEMES[num_train_prompts : num_train_prompts + num_test_prompts]
    test_id = 1
    for prompt_id, prompt_text in test_prompts:
        topic_short = prompt_text.split()[1:4]
        topic_str = " ".join(topic_short)
        for model in TARGET_MODELS:
            template = MODEL_STYLE_FRAGMENTS[model]
            body = template.format(prompt_topic=topic_str)
            test_rows.append(
                {
                    "id": f"test_{test_id:04d}",
                    "prompt_id": prompt_id,
                    "text": body,
                }
            )
            test_id += 1

    df_test = pd.DataFrame(test_rows)
    test_path = SAMPLE_DATA_DIR / "test.csv"
    df_test.to_csv(test_path, index=False)

    console.print(
        f"[bold green]✓ Created sample dataset in {SAMPLE_DATA_DIR}:[/bold green]\n"
        f"  - train.csv: {len(df_train)} rows across {len(train_prompts)} prompts and {len(TARGET_MODELS)} models\n"
        f"  - test.csv : {len(df_test)} rows across {len(test_prompts)} unseen test prompts"
    )


if __name__ == "__main__":
    generate_sample_data()
