# QA-MAB vs NB3R — ניתוח עומק מבוסס קוד

## 1. הסביבה — simulation_core.py

### מה האלגוריתמים לא רואים (Ground Truth)
- **B[i,k]** ∈ [0.5, 1.0] — תועלת בסיסית. shape: (N, m)
- **I[i,k,j,l]** ∈ [0, 0.2] — הפרעה. shape: (N, m, N, m). **לא סימטרי!**
- I[i,:,i,:] = 0 (אין הפרעה עצמית)

### מה האלגוריתמים כן רואים
רק throughput סופי אחרי כל step:

`U_i = B[i, k_i] − Σ_{j,l} I[i, k_i, j, l] · x[j, l]`

סקלר אחד per agent. לא יודעים מה B ומה I בנפרד.

---

## 2. NB3R — האלגוריתם המבוזר

### מבני נתונים
- **W[i,k]**: shape (N, m), מאותחל **אפסים**
- **τ**: מתחיל 0.1, עולה 0.05 per step

### Step מלא

#### שלב 1: בחירת מסלולים (Softmax)
כל סוכן בוחר מסלול עם:

`P_{i,k} = exp(τ · W[i,k]) / Σ_l exp(τ · W[i,l])`

- ב-t=0: W=0 → P אחיד (1/m) → exploration מלא
- ב-t=100: τ=5.1, W שונים → exploitation מתחיל
- ב-t=500: τ=25.1 → כמעט one-hot (greedy)

#### שלב 2: מדידת throughput
`throughputs = env.compute_throughput(chosen_routes)`

כל סוכן מקבל U_i — סקלר אחד.

#### שלב 3: שידור לשכנים
כל סוכן רואה את U_j של **כל** N-1 השכנים (fully connected).

#### שלב 4: עדכון משקלים ← **הצעד הקריטי**
```
total_signal = U_i + Σ_{j≠i} U_j = Social Welfare הכולל
W[i, k_chosen] ← (1-α)·W[i, k_chosen] + α·total_signal
```

**מה קורה פה:**
- רק המסלול שנבחר מתעדכן. השאר נשארים כפי שהם.
- total_signal = SW הכולל, **זהה לכל הסוכנים** באותו step
- W לומד: "כשבחרתי מסלול k, ה-SW הכולל היה X"
- אם מסלול k גורם הרבה interference → SW נמוך → W[i,k] יורד
- אם מסלול k נמנע מinterference → SW גבוה → W[i,k] עולה

**למה זה עובד:** NB3R לומד correlation ישירה בין "מה בחרתי" ל-"כמה טוב היה לכולם". לא צריך להפריד B מ-I. ה-SW *כבר* מקודד את כל המידע.

#### שלב 5: τ עולה
`τ ← τ + 0.05`

#### סדרי גודל
- Parameters: N×m = 40 (for N=10, m=4)
- Per step: O(N²) — שידור + עדכון
- **אין solver, אין אופטימיזציה**

---

## 3. QA-MAB — האלגוריתם הריכוזי

### מבני נתונים
- **u_hat[i,k]**: shape (N, m), מאותחל **0.75** (אמצע [0.5,1])
- **I_hat[i,k,j,l]**: shape (N, m, N, m), מאותחל **אפסים**
- **τ**: מתחיל 0.1, עולה 0.05 per step

### Step מלא

#### שלב 1a: בניית QUBO
מטריצה Q בגודל (N·m × N·m) = (40 × 40) for N=10, m=4:

**אלכסון:**
`Q[idx(i,k), idx(i,k)] = -u_hat[i,k] - λ/2`

**Off-diagonal same-agent (אילוץ one-hot):**
`Q[idx(i,k), idx(i,l)] = λ/2` (for k≠l, same i)

**Off-diagonal cross-agent (interference):**
`Q[idx(i,k), idx(j,l)] = I_hat[i,k,j,l]` (for i≠j)

**Scaling:**
`Q_A = τ · Q`

**מה ה-QUBO מנסה לעשות:**

`min_x  x^T Q_A x = τ · [−Σ u_hat · x + Σ I_hat · x·x' + λ · Σ (Σ_k x_{i,k} − 1)²]`

כלומר: **למקסם u_hat** (מסלולים טובים), **למזער I_hat** (הפרעות), **לכפות** מסלול אחד per agent.

#### שלב 1b: פתרון QUBO עם SA

SA parameters (for N≤10): 8 restarts × 500 iterations.

אתחול: greedy `x[i, argmax(u_hat[i])] = 1` (restarts נוספים עם perturbation).

Per iteration:
1. בחר סוכן אקראי i
2. החלף מסלול: k_old → k_new (אקראי, שונה מk_old)
3. חשב ΔE = E(x_new) - E(x_old)
4. קבל אם ΔE < 0 או random < exp(-ΔE/T_sa)
5. T_sa *= 0.9

**Total SA work per step:** 8 × 500 = 4,000 iterations, כל אחד O(N·m) → ~O(160K) ops.

#### שלב 2: מדידת throughput
`throughputs = env.compute_throughput(assignment)`

#### שלב 3a: עדכון u_hat ← **בעיה ראשונה**
```
u_hat[i,k] ← u_hat[i,k] + 0.2 · (observed_i − u_hat[i,k])
```

**u_hat לומד את observed = B[i,k] − interference, לא את B[i,k] עצמו.**
ה-QUBO משתמש ב-u_hat כ-"utility" ומוסיף I_hat כ-"penalty".
זו ספירה כפולה חלקית — documented as mean-field approximation.

**בעיה:** u_hat משתנה תלוי ב-interference שהיה באותו step. אם הinterference היה גבוה, u_hat יורד, מה שגורם ל-QUBO "לשנוא" את המסלול — גם אם B שלו גבוה.

#### שלב 3b: למידת I_hat ← **בעיה שנייה**

עובד על **הstep הקודם** (_prev_x, _prev_throughputs):

```
for כל זוג (i,j) where j > i:
    ki, kj = מסלולים מהstep הקודם
    drop_i = max(0, u_hat[i,ki] − prev_throughputs[i])
    drop_j = max(0, u_hat[j,kj] − prev_throughputs[j])
    
    if drop_i > 0.02:
        I_hat[i,ki,j,kj] += 0.05  (capped at 0.3)
    if drop_j > 0.02:
        I_hat[j,kj,i,ki] += 0.05  (capped at 0.3)
```

**בעיות קריטיות:**

1. **I_hat רק עולה, אף פעם לא יורד.** אם collision הוא false positive (ה-drop בגלל סוכן שלישי), I_hat מנופח לנצח.

2. **Drop מבוסס על u_hat שמשתנה.** ב-step 3a עדכנו u_hat, ואז משתמשים בו כ-expected. אם u_hat ירד בגלל update → drop קטן → פחות learning. Feedback loop מוזר.

3. **לומד רק זוגות שנבחרו באותו step.** עם m=4, P(שני סוכנים ספציפיים בוחרים זוג מסלולים ספציפי) = 1/16. צריך ~16 steps לכל observation. ויש 720 זוגות → צריך ~11,520 observations → בערך T=700 steps **minimum** (best case).

4. **"אשמה קולקטיבית"** — אם agent i חווה drop כשgent j בחר route l, I_hat[i,ki,j,kj] עולה. אבל אולי agent k (שלישי) גרם ל-drop. אין הפרדה.

#### שלב 4: τ עולה
`τ ← τ + 0.05` (זהה ל-NB3R)

#### סדרי גודל
- Parameters: N×m + N×m×N×m = 40 + 6,400 = **6,440** (for N=10, m=4)
- Per step: O(N²m² + restarts × iters × N·m) ≈ O(166K)
- **SA solver dominates runtime**

---

## 4. ההבדלים שמשפיעים — ניתוח מעמיק

### 4.1 Learning Signal

**NB3R:** לומד `total_signal = SW` — סיגנל **חזק ויציב**.
- כל step מעדכן (גם אם הייתה התנגשות)
- ה-EMA מחליק רעש
- W[i,k] מתכנס ל-"expected SW when I choose k"

**QA-MAB:** לומד `observed = U_i` + `collision detection` — סיגנלים **חלשים ורועשים**.
- u_hat מקבל observed שמשתנה wildly בין steps (תלוי מה כולם בחרו)
- I_hat מתעדכן רק ב-~6% מהsteps (1/16 probability per pair)
- Feedback loops בין u_hat ו-I_hat

### 4.2 מהירות התכנסות

**NB3R:** W[i,k] שנבחר פעם אחת כבר מקבל signal.
- אחרי ~4m steps (16 for m=4), כל מסלול נדגם ~4 פעמים → W מתחיל להבדיל
- אחרי ~50 steps, τ=2.6 → softmax מתחיל לבחור טוב

**QA-MAB:** u_hat מתכנס אחרי ~50 steps per route (B_learn_rate=0.2).
- אבל I_hat צריך ~700+ steps כדי לראות כל זוג
- QUBO quality ≈ u_hat quality × I_hat quality
- אם אחד מהם גרוע, ה-SA פותר בעיה שגויה

### 4.3 Solver Quality

**NB3R:** אין solver. Softmax **הוא** ה-solver — כל סוכן פועל באופן עצמאי, ההתנהגות הקולקטיבית מתכנסת (Nash-like).

**QA-MAB:** SA solver עם 4,000 iterations. על QUBO של 40 variables זה **מספיק** כשה-QUBO מדויק. אבל אם I_hat=0 (steps ראשונים), ה-QUBO הוא `min -u_hat` → SA פשוט בוחר greedy(u_hat) → **מתעלם מinterference לחלוטין**.

### 4.4 למה QA-MAB מנצח ב-T ארוך

ב-T=10,000:
1. I_hat מתכנס (מספיק observations לכל זוג)
2. u_hat מתכנס ל-effective utility
3. QUBO מייצג את הבעיה נאמנה
4. SA מוצא פתרון גלובלי טוב (כי ה-QUBO נכון)
5. **QA-MAB רואה את כל ה-interference pattern גלובלית** ← NB3R לא יכול

ב-T=500:
1. I_hat עדיין רועש ולא שלם
2. u_hat עדיין לא יציב
3. QUBO שגוי → SA פותר בעיה שגויה
4. NB3R כבר התכנס עם W

### 4.5 החולשה המבנית של QA-MAB(SA)

גם ב-T ארוך, SA **נתקע ב-local minima**. הNוף האנרגטי של QUBO עם τ גבוה מלא מחסומים.

**NB3R לא סובל מזה** כי softmax תמיד שומר הסתברות > 0 לכל מסלול (exploration מובנה).

**QA אמיתי פותר את זה** עם quantum tunneling — חודר דרך מחסומים במקום לטפס מעליהם.

---

## 5. מסקנות מעשיות

### מה צריך לתקן ב-QA-MAB
1. **I_hat decay** — אפשר ל-I_hat לרדת, לא רק לעלות
2. **UCB exploration** — u_hat + confidence bonus לmסלולים שנדגמו מעט
3. **Warm-start SA** — להתחיל מהפתרון הקודם (כמו cyclic annealing)
4. **Batch observations** — ללמוד I_hat מmultiple steps בו-זמנית

### מה QA אמיתי יפתור
- **שלב 1b (solver)** — tunneling במקום SA → פתרון global
- **לא** שלבים 3a/3b (learning) — עדיין צריך ללמוד u_hat ו-I_hat מthroughputs

### הטיעון לתזה
> ה-framework של QA-MAB נכון. הבעיה היא ב-**solver** (SA) ובמהירות ה-**learning** (I_hat).
> QA אמיתי פותר את בעיית ה-solver.
> Learning improvements (decay, UCB, warm-start) פותרים את בעיית הלמידה.
> שניהם ביחד → QA-MAB dominant.

---

## Links
- [[qa-mab-simulation-results]] — תוצאות סימולציה
- [[qa-mab-hardware-analysis]] — ניתוח חומרת D-Wave
- [[DIAMOND-paper-notes]] — המאמר המקורי

## Tags
#thesis #quantum #algorithm-analysis #qa-mab #nb3r #deep-dive
