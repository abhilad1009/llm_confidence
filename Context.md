There and back again and again. Probably

This is a very solid idea — you’re essentially turning a generative LLM into a discriminator-style extractor by stress-testing invariances and sensitivities.

You’re approximating confidence through causal perturbation testing, which is much more principled than naive log-prob averaging.

🧪 What This Gives You
Property	Traditional OCR	Your LLM System
Localization	Bounding box	Anchor paragraph
Value robustness	Detection score	Substitution invariance
Context sensitivity	Classifier probability	Anchor corruption sensitivity
Confidence	Softmax score	Empirical perturbation score
🧠 Why This Works

You're estimating:

𝑃
(
correct extraction
∣
anchor stability
,
value localization
)
P(correct extraction∣anchor stability,value localization)

Instead of trusting internal logits, you approximate causal reliability.

This is closer to:

Conformal prediction

Causal intervention testing

Counterfactual validation

🔬 Advanced Improvements (Production Grade)

If you want this to be research-grade:

Use attention rollout for real anchor detection

Use gradient attribution for token importance

Replace heuristic corruption with masked language modeling

Bootstrap confidence intervals

Use entropy of output logits

Combine with Monte Carlo dropout

🏆 Key Insight

You are effectively converting:

Generative LLM → Empirical probabilistic extractor

Without retraining.