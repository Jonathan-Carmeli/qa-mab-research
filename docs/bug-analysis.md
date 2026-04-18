# QA-MAB Opus Deep Analysis — למה QA-MAB(SA) נכשל ואיך לתקן

## אבחנה מרכזית: Double-Counting

ה-"mean-field decomposition" שתועד בקוד **שגוי מתמטית**.

u_hat לומד `observed = B - I_actual`. ה-QUBO עושה:
```
E = -u_hat·x + I_hat·x·x'
  ≈ -(B - I)·x + I_hat·x·x'
  ≈ -B·x + I·x + I_hat·x·x'
  = -B·x + 2·I·x·x'   (interference counted TWICE)
```

**אין פרשנות mean-field שבה זה תקין.** אם u_hat → B אז ה-QUBO נכון. אם u_hat → B-I, הinterference נספר פעמיים.

## 6 באגים/פגמים (לפי חומרה)

### 🔴 FATAL 1: Double-Counting Interference
u_hat = B - I_actual. QUBO מוסיף I_hat. סה"כ 2×I.
**תיקון:** u_hat ילמד B ע"י `B_est = observed + Σ I_hat`

### 🔴 FATAL 2: I_hat Over-Attribution (×9)
כשסוכן i חווה drop, **כל 9 השכנים** מקבלים +0.05 ל-I_hat. True contribution of each = drop/9. אבל מזכים כל אחד ב-full drop → ×9 inflation.
**ראיה:** I_hat mean = 0.254 vs true 0.090 (×2.8 ≈ 9 normalized)

### 🔴 FATAL 3: No Exploration Mechanism
אין UCB, אין Thompson Sampling, אין ε-greedy. ב-τ>5 ה-SA greedy לחלוטין. Assignment קבוע → אין observations חדשות → learning מת.

### 🟡 MAJOR 4: τ Growth Kills Learning
ב-t=500: τ=25. Metropolis acceptance exp(-τΔQ/T_SA) ≈ 0 לכל ΔQ>0. Assignment deterministic.
**תיקון:** `tau = min(tau + delta_tau, 5.0)`

### 🟡 MAJOR 5: Drop Threshold (0.02) Below Noise Floor
Typical noise ~0.1 per step. Threshold 0.02 fires almost always → I_hat inflates rapidly.
**תיקון:** `threshold = 2 * std(recent_throughputs)`

### 🟡 MAJOR 6: Circular Dependency in Drop Detection
`expected = u_hat[i,ki]` (just updated toward observed) → after a few steps, expected ≈ observed → drop ≈ 0 → I_hat stops updating. Self-consistency kills learning.

## תיקונים מוצעים (7 fixes, סדר עדיפות)

### Fix #1: u_hat → B (לא observed)
```python
for i in range(self.N):
    k = assignment[i]
    I_est = sum(self.I_hat[i, k, j, assignment[j]]
                for j in range(self.N) if j != i)
    b_est = throughputs[i] + I_est
    self.u_hat[i, k] += self.B_learn_rate * (b_est - self.u_hat[i, k])
```

### Fix #2: Attribution Weighted (לא blame-all)
```python
# Weight by current I_hat estimates
total_I = sum(I_hat[i,ki,j,kj] for j != i)
for j != i:
    weight = I_hat[i,ki,j,kj] / (total_I + eps)
    I_hat[i,ki,j,kj] += lr * weight * drop_i
```

### Fix #3: UCB Exploration Bonus
```python
ucb_bonus = c * np.sqrt(np.log(self.t + 1) / (self.visit_count + 1))
# In QUBO diagonal: -(u_hat + ucb_bonus) - lambda/2
```

### Fix #4: τ Cap
```python
self.tau = min(self.tau + self.delta_tau, 5.0)
```

### Fix #5: Statistical Threshold
```python
threshold = 2.0 * np.std(self.throughput_history[i][-20:])
```

### Fix #6: I_hat Decay
```python
self.I_hat *= (1 - 0.001)  # slow decay every step
```

### Fix #7: Lower I_cap
```python
I_cap = 0.2  # match I_max of moderate environment
```

## האם ה-Framework ניתן להצלה?

**כן. חד-משמעית.**

מבנה ה-QUBO נכון מתמטית:
```
min  -Σ B[i,k] x_ik + Σ_{i≠j} I[i,k,j,l] x_ik x_jl + λ·Σ(Σ_k x_ik - 1)²
```

הכשל הוא **רק** ב-learning של u_hat ו-I_hat. הQUBO structure = social welfare objective.

### מה צפוי אחרי תיקונים:
- **Fix #1 + #4**: u_hat מתכנס ל-B, τ capped → SA ממשיך לחקור → QA-MAB(SA) אמור לנצח NB3R ב-T=500
- **Fix #1 + #2 + #3**: learning מדויק + exploration → QA-MAB(BF) dominant כבר ב-T=200
- **QA אמיתי + all fixes**: tunneling + accurate model → dominant בכל N

### תובנה לתזה:
> אם אחרי Fix #1+#4 QA-MAB **כן** מנצח → הראינו ש-**model accuracy** היא ה-bottleneck, לא ה-framework.
> אם עדיין לא מנצח → "centralized with imperfect learning loses to distributed direct feedback" — **גם זו תוצאה מפרסמת.**

## Links
- [[qa-mab-T100k-analysis]] — תוצאות T=100K
- [[qa-mab-algorithm-deep-dive]] — ניתוח קוד מפורט
- [[qa-mab-simulation-results]] — כל הניסויים

## Tags
#thesis #quantum #bugs #fixes #opus-analysis #critical
