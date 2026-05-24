# Simulated Annealing — Route-Flip Variant

## מה הבעיה שה-SA פותר?

בכל צעד זמן, ה-agent צריך לבחור נתיב `k_n ∈ {0, …, K-1}` לכל flow `n ∈ {0, …, N-1}`.
הבחירה מנוסחת כבעיית QUBO: מצא וקטור בינארי `x ∈ {0,1}^M` (כאשר `M = N·K`)
שממזער `E = x^T Q x`, תחת אילוץ one-hot: בדיוק ביט אחד דולק לכל flow.

## למה route-flip ולא bit-flip?

ב-SA רגיל (bit-flip) מציעים לדלק/לכבות ביט בודד. זה יכול לשבור את אילוץ ה-one-hot —
כגון לקבל 0 ביטים דולקים לflow מסוים, או 2. כדי לתקן זאת נאלצים להוסיף penalty
גדול ל-Q. ב-route-flip מציעים תמיד **להחליף נתיב שלם**: flow `n` עובר מנתיב `k_old`
לנתיב `k_new ≠ k_old`. כך one-hot מובטח **מבנית** בכל צעד, ואין צורך ב-penalty.

---

## האלגוריתם — שלב אחר שלב

### ייצוג

```
active[n] = n·K + k_n      (אינדקס גלובלי ב-QUBO)
```

כלומר `active` הוא וקטור `(N,)` שמאחסן את האינדקס הגלובלי של הנתיב הנבחר לכל flow.
האנרגיה הכוללת היא `E = Σ_{i,j} Q[active[i], active[j]]`.

### מטמון O(1): Q_row_sum ו-Q_col_sum

```python
Q_row_sum[j] = Σ_n  Q[j, active[n]]        # עמודה j, שורות active בלבד
Q_col_sum[j] = Σ_n  Q[active[n], j]        # שורה active[n], עמודה j
energy       = Σ_n  Q_col_sum[active[n]]   # = x^T Q x
```

(עבור Q סימטרי: `Q_row_sum = Q_col_sum`, אבל מחשבים שניהם לדיוק.)

### delta-E כאשר active[n]: old_idx → new_idx

```
ΔE = (Q[new,new] − Q[old,old])
   + Σ_{l≠n} (Q[new, active[l]] − Q[old, active[l]])   ← sum_row
   + Σ_{l≠n} (Q[active[l], new] − Q[active[l], old])   ← sum_col
```

בקוד (O(1) עם המטמון):
```python
sum_row = (Q_row_sum[new_idx] - Q[new_idx, old_idx]) - (Q_row_sum[old_idx] - Q[old_idx, old_idx])
sum_col = (Q_col_sum[new_idx] - Q[old_idx, new_idx]) - (Q_col_sum[old_idx] - Q[old_idx, old_idx])
delta   = (Q_diag[new_idx] - Q_diag[old_idx]) + sum_row + sum_col
```

### קריטריון מטרופוליס

```python
if delta < 0 or (T > 1e-10 and rng.random() < exp(-delta / T)):
    # accept flip
```

- אם ΔE < 0: תמיד מקבלים (מצב טוב יותר).
- אחרת: מקבלים בהסתברות `exp(-ΔE/T)`. כשT גבוהה — מקבלים כמעט הכל (exploration).
  כשT נמוכה — מקבלים רק שיפורים (exploitation).

### עדכון המטמון אחרי קבלה

```python
Q_row_sum += Q[:, new_idx] - Q[:, old_idx]
Q_col_sum += Q[new_idx, :] - Q[old_idx, :]
```

עלות: O(M) — הוספת שני וקטורי עמודה/שורה.

### לוח טמפרטורות

**ירידה גיאומטרית**: `T ← T · decay` בכל צעד (decay = 0.999 כברירת מחדל).

כל restart מתחיל בטמפרטורה גבוהה יותר:
```python
T = T0 * (1.0 + restart * 0.3)
```
כך restarts מאוחרים מסייחים את הפתרון מהמינימום המקומי של הrestart הקודם.

---

## פסאודו-קוד מלא

```
קלט: Q (M×M), N, K, n_restarts, n_iters, T0, decay, rng
פלט: chosen (N,) — אינדקסי נתיב לכל flow

best_active ← None,  best_energy ← ∞

לכל restart r = 0..n_restarts-1:
    active[n] ← n·K + randint(0,K)   לכל n     ← אתחול רנדומלי
    חשב Q_row_sum, Q_col_sum, energy
    עדכן best אם energy < best_energy

    T ← T0 · (1 + r·0.3)             ← טמפרטורת פתיחה

    לכל step s = 0..n_iters-1:
        T ← T · decay                 ← קירור גיאומטרי
        בחר n ~ Uniform{0..N-1}
        בחר k_shift ~ Uniform{1..K-1} → k_new = (k_old + k_shift) mod K
        חשב delta = ΔE               ← O(1)
        אם delta<0 או rand()<exp(-delta/T):
            עדכן active[n] ← new_idx
            עדכן Q_row_sum, Q_col_sum  ← O(M)
            עדכן best אם energy < best_energy

החזר [best_active[n] - n·K לכל n]
```

---

## השוואה ל-SA הישן (bit-flip)

| | Bit-flip (ישן) | Route-flip (חדש) |
|---|---|---|
| **ייצוג** | `x ∈ {0,1}^(N·K)` | `active[n] ∈ [0,K-1]` |
| **הצעה** | flip ביט בודד | החלף נתיב שלם |
| **one-hot** | לא מובטח → penalty | מובטח מבנית |
| **לוח T** | ליניארי (`linspace`) | גיאומטרי (`T *= decay`) |
| **מטמון** | `h = Q@x` | `Q_row_sum`, `Q_col_sum` |
| **restarts** | 20 | 20 (default) |
| **צעדים/restart** | 200 sweeps × M steps | 1000 iters |

---

## פרמטרים (ב-`QAMABPhysical`)

| פרמטר | ברירת מחדל | משמעות |
|---|---|---|
| `sa_n_restarts` | 20 | מספר ריצות SA עצמאיות |
| `sa_n_iters` | 1000 | צעדי route-flip לכל restart |
| `sa_T0` | 2.0 | טמפרטורת פתיחה (לrestart הראשון) |
| `sa_decay` | 0.999 | מקדם קירור גיאומטרי |

---

## איך ה-SA משתלב ב-QA-MAB

```
act(t, p):
    Q = build_qubo()                    ← בנה מטריצת QUBO מ-θ̂, φ̂
    gamma = γ₀ / ((p+1)^a · (t+1)^b)   ← טמפרטורת exploration-exploitation
    Q_scaled = Q / gamma                ← QUBO חד יותר = פחות exploration
    chosen = sa_solve(Q_scaled, N, K, rng, ...)
    return chosen
```

כשgamma גדולה (מוקדם) → Q_scaled קטן → SA רואה מינימום שטוח → יותר exploration.
כשgamma קטנה (מאוחר) → Q_scaled גדול → SA רואה מינימום חד → יותר exploitation.
