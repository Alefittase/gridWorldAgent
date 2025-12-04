## **(a) Do value iteration and policy iteration converge to the same answer?**

### **Deterministic version**

In the deterministic environment, both value iteration and policy iteration converged to the **same optimal policy** and therefore produced the same optimal path:

```
(0,0) → (0,1) → (0,2) → (1,2) → (1,3) → (2,3) → (3,3) → (3,4)
```

So in the deterministic MDP, **both algorithms agree**, and the optimal policy is stable.

---

### **Stochastic version (0.7 intended action, 0.1 for the other three)**

In the stochastic environment, **value iteration and policy iteration still agree with each other**, but their resulting policy is **different from the deterministic optimal policy**.

The stochastic optimal policies use more **downward and left-side actions** in places where the deterministic policy preferred going right or upward.
This happens because the agent must consider sideways slips, and therefore some previously optimal states become riskier.

So the final answer is:

* **Value iteration and policy iteration still match each other**
* **But the stochastic-optimal policy is NOT the same as the deterministic-optimal policy**

Thus, the environment’s stochasticity **changed the optimal policy**.

---

## **(b) Is the optimal policy unique?**

### Deterministic:

Yes — deterministic runs produced **one single unique optimal policy** across all γ values and across both algorithms.
All 100 runs per setting resulted in identical policy grids.

### Stochastic:

Yes again — in the stochastic experiments, the policy grids were also **identical across all runs** and across both algorithms and all γ values.

However:

* **The deterministic optimal policy ≠ the stochastic optimal policy**,
  but **within each environment type**, the optimal policy was unique.

---

## **(c) Which algorithm is faster?**

From both files:

### Value iteration:

* Required **more iterations** (typically ~7–9 deterministic, ~8 stochastic)
* Slower in runtime (about **2–3× slower**, depending on γ and environment)

### Policy iteration:

* Required only **2–3 policy updates**, plus a few evaluation sweeps
* Significantly faster runtime

### Conclusion:

Policy iteration is **consistently faster** than value iteration in both deterministic and stochastic environments.

---

## **(d) How does γ affect the results?**

Across both models:

### Low γ (0.1):

* The agent is short-sighted
* Values remain small
* Convergence is fastest
* Policy does not change in either model

### Moderate γ (0.6):

* Values increase
* More iterations needed
* Policy remains the same

### High γ (0.9):

* Values become significantly larger
* Convergence takes longest
* Policy again stays the same

### Important:

Within each environment type (deterministic or stochastic), **γ never changed the policy**.
It only affected:

* convergence speed
* the shape and magnitude of the value function

---

## **(e) What changes when transitions become stochastic?**

This is where the most important difference appears.

### Observed differences:

1. **The optimal policy changed** between the deterministic and stochastic models.
2. The stochastic policy used **more downward actions** and avoided some states the deterministic policy considered safe.
3. Value tables changed significantly — especially around states near obstacles, where stochastic sideways drift increases risk.
4. Convergence slowed because each action involves four transition outcomes.

### Interpretation:

Stochasticity forces the agent to plan more conservatively.
Actions that were safe in the deterministic grid may become risky when there is a **30% chance of drifting into obstacles or walls**.

As a result:

* The stochastic agent chooses **safer, more stable routes**
* The deterministic shortest path is no longer optimal when slips are possible