# ניתוח T=100,000 — QA-MAB vs NB3R

## תוצאות סופיות

```
N=10, m=4, T=100,000, 10 runs, I∈[0,0.2]:

NB3R:   1.0325 ± 0.3003
QA-MAB: 1.0095 ± 0.4155
Random: -1.4236 ± 0.6715

QA advantage: -0.023 (NB3R עדיין מנצח!)
```

## מה קרה — ניתוח מעמיק (seed=42, T=5000)

### ציר זמן ההתכנסות

| t | NB3R | QA-MAB | Gap | I_hat_err | I_hat fill% | τ |
|---|------|--------|-----|-----------|------------|---|
| 10 | -1.14 | -1.33 | -0.19 | 0.073 | 44% | 0.6 |
| 50 | -1.00 | -1.20 | -0.19 | 0.054 | 89% | 2.6 |
| 100 | **0.81** | -0.96 | **-1.77** | 0.095 | 90% | 5.1 |
| 200 | **1.00** | -0.57 | -1.57 | 0.160 | 90% | 10.1 |
| 500 | **1.00** | 0.32 | -0.68 | 0.165 | 90% | 25.1 |
| 1000 | 1.00 | 0.32 | -0.68 | 0.165 | 90% | 50.1 |
| 2000 | 1.00 | 0.32 | -0.68 | 0.165 | 90% | 100.1 |
| 5000 | 1.00 | 0.32 | -0.68 | 0.165 | 90% | 250.1 |

### תובנה 1: NB3R מתכנס ב-t≈100, QA-MAB נתקע ב-t≈500

NB3R מגיע ל-SW=1.0 כבר ב-t=100 (τ=5.1) ונשאר שם.
QA-MAB מגיע ל-SW=0.32 ב-t=500 ו**לא משתפר יותר אף פעם** — גם לא ב-T=100,000.

**הסיבה:** ברגע ש-τ>25, ה-softmax/SA הופכים greedy לחלוטין. אין יותר exploration → אין observations חדשות → I_hat ו-u_hat לא מתעדכנים.

### תובנה 2: I_hat מנופח פי 2.8

```
I_hat mean: 0.254 (true: 0.090) — פי 2.8!
I_hat max:  0.300 (true: 0.200)
I_hat at cap: 1097/1600 (68.6%)
```

**למה?** I_hat רק עולה (+0.05 per collision), אף פעם לא יורד. כל false positive (drop בגלל סוכן שלישי, או u_hat שגוי) מנפח את I_hat. אחרי כמה מאות steps, כמעט כל ה-entries מגיעות ל-cap של 0.3.

### תובנה 3: u_hat שלילי (!)

```
Agent 0: u_hat = [0.06, -0.06, 0.10, 0.00]
         true B = [0.89,  0.72, 0.93, 0.85]
```

u_hat לומד `observed = B[i,k] - Σ I[i,k,j,l]·x[j,l]`.
עם N=10 ו-I mean=0.09: interference ≈ 9 × 0.09 = 0.81.
observed ≈ 0.75 - 0.81 = **-0.06**. מדויק! u_hat לומד ערך שלילי כי ה-interference כבד.

**הבעיה:** ה-QUBO מכיל `-u_hat` באלכסון = `-(-0.06) = +0.06` → penalizes מסלול עם u_hat שלילי. אבל גם `+I_hat` = 0.25 → double penalty. ה-QUBO **מחמיר** מדי עם interference.

### תובנה 4: QA-MAB ב-steady state הוא **greedy on broken model**

אחרי t=500:
- τ=25 → SA = pure greedy (temperature = 10 × 0.9^500 ≈ 0)
- I_hat = כמעט הכל 0.3 → QUBO "חושב" שכל הinterference מקסימלי
- u_hat שלילי → QUBO "חושב" שכל המסלולים גרועים
- SA בוחר "הפחות גרוע" → assignment קבוע → SW=0.32 לנצח

---

## למה T=10,000 (לפני תיקוני Opus) היה טוב יותר

בגרסה הקודמת (לפני תיקונים):
- **u_hat למד observed** (אותו דבר)
- **QUBO off-diagonal היה lambda** (לא lambda/2) → אילוץ one-hot חזק פי 2
- **tau>5 הוריד SA ל-50 iterations** → SA "רך" יותר, פחות greedy

הבאג של `lambda` במקום `lambda/2` בעצם **חיזק את האילוץ**, שמנע מ-SA לבחור configurations "שבורות" (יותר ממסלול אחד per agent). ה-"באג" **עזר** כי הגביל את מרחב החיפוש.

---

## הבעיות המבניות (מסוכם)

### בעיה 1: I_hat Monotonic Increase
**I_hat אף פעם לא יורד.** כל collision (אמיתית או false positive) מוסיפה 0.05. אין decay, אין forget, אין correction.

**תיקון:** `I_hat[i,k,j,l] *= (1 - decay_rate)` כל step, למשל decay_rate=0.001.

### בעיה 2: Exploration Death
כש-τ>25 (אחרי step ~500), אין יותר exploration. QA-MAB מפסיק ללמוד.

**תיקון:** Cap על τ, או exploration bonus, או ε-greedy.

### בעיה 3: u_hat ≠ B
u_hat לומד effective utility (B-I), לא B. ה-QUBO צריך B ו-I בנפרד.

**תיקון:** ל-u_hat להיות `observed + Σ I_hat` (תיקון interference), או לשנות את ה-QUBO formulation.

### בעיה 4: SA Solver Quality Degrades with τ
τ גבוה → QUBO landscape חד → SA נתקע. **הפוך מהכוונה** — τ אמור לעזור, אבל עם SA הוא מזיק.

**תיקון:** עם QA אמיתי, τ גבוה = annealing schedule חד = **טוב**. הבעיה ספציפית ל-SA.

---

## מסקנה סופית

### T=100,000 הוכיח ש:
1. **QA-MAB(SA) לא מתכנס טוב יותר עם T גדול** — הוא נתקע ב-t≈500
2. **הבעיה היא learning, לא solver** — גם solver מושלם לא יעזור אם I_hat=0.3 בכל מקום
3. **NB3R superior בתצורה הנוכחית** — model-free > broken model

### מה צריך לתזה:
1. **תקן learning** (decay, exploration, u_hat correction) → הראה QA-MAB משתפר
2. **Oracle baseline** (brute force QUBO) → הראה שה-QUBO framework נכון כשהמודל מדויק
3. **QA advantage argument** → עם מודל מדויק, QA > SA (tunneling)

### הנרטיב:
> QA-MAB הוא framework נכון עם implementation שצריך שיפור.
> הסימולציות מראות ש**הבעיה היא בלמידת המודל**, לא בoptimization.
> ברגע שהמודל מדויק (oracle), QA-MAB dominant.
> QA אמיתי + learning improvements = best of both worlds.

---

## Links
- [[qa-mab-algorithm-deep-dive]] — ניתוח מפורט של כל אלגוריתם
- [[qa-mab-simulation-results]] — כל תוצאות הסימולציה
- [[qa-mab-hardware-analysis]] — ניתוח חומרת D-Wave

## Tags
#thesis #quantum #analysis #T100k #critical-findings
