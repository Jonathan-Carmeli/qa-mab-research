# QA-MAB Hardware Analysis: D-Wave Capability & Future Projections

## הבעיה שלנו
QUBO עם N×m משתנים בינאריים. שני סוגי קשרים:
- **Same-agent constraint**: m*(m-1)/2 קשרים per agent (dense, אבל קטן)
- **Cross-agent interference**: I_hat[i,k,j,l] > 0 רק בין שכנים ברשת

### Sparsity של ה-QUBO
ברשת תקשורת אמיתית, כל סוכן מפריע רק ל-d שכנים (d = degree הרשת).
- **Fully connected** (worst case): O(N²m²) קשרים → צריך clique embedding
- **Sparse** (רשת עם degree d): O(N·d·m²) קשרים → embedding יעיל הרבה יותר

---

## היום: D-Wave Advantage2 (2025-2026)

### מפרט
- **4,400+ qubits**, Zephyr topology, 20-way connectivity
- **40,000+ couplers**
- Anneal time: **20μs** per sample
- Coherence: ×2 מ-Advantage

### מה אפשר לפתור

#### Fully Connected QUBO (הסימולציה שלנו)

| N | m | Variables | Physical qubits (est.) | אפשרי? | זמן per step |
|---|---|-----------|----------------------|---------|-------------|
| 5 | 4 | 20 | ~80 | ✅ | ~20μs |
| 10 | 4 | 40 | ~320 | ✅ | ~20μs |
| 15 | 4 | 60 | ~900 | ✅ | ~20μs |
| **17** | **4** | **68** | **~1,500** | ✅ **max** | ~20μs |
| 20 | 4 | 80 | ~2,500 | ⚠️ chains ארוכים, איכות יורדת | ~20μs |
| 25 | 4 | 100 | ~4,000+ | ❌ לא נכנס | — |

**Clique embedding formula**: K_n על Zephyr דורש ~n²/20 physical qubits (הערכה).
4,400 qubits → K_~68 → **N_max ≈ 17 עם m=4**.

#### Sparse QUBO (רשת עם degree d=4-6)

| N | m | d | Variables | Edges | Physical qubits (est.) | אפשרי? |
|---|---|---|-----------|-------|----------------------|---------|
| 20 | 4 | 4 | 80 | ~640 | ~200 | ✅ |
| 50 | 4 | 4 | 200 | ~1,600 | ~600 | ✅ |
| 100 | 4 | 6 | 400 | ~4,800 | ~1,500 | ✅ |
| 200 | 4 | 6 | 800 | ~9,600 | ~3,500 | ⚠️ בגבול |
| 300 | 4 | 4 | 1,200 | ~9,600 | ~4,000 | ⚠️ tight |

**Sparse embedding**: ~2-4 physical qubits per logical qubit (vs ~n/20 for clique).

### השוואת סיבוכיות זמן — היום

| Solver | זמן per QUBO solve | עבור N=10, m=4 |
|--------|-------------------|----------------|
| **SA** (8 restarts × 500 iters) | O(restarts × iters × N·m) | ~15ms |
| **D-Wave QA** (1 sample) | O(1) — קבוע! ~20μs anneal | ~20μs + ~10ms communication |
| **D-Wave QA** (100 samples) | 100 × 20μs = 2ms | ~2ms + ~10ms communication |
| **Brute Force** | O(m^N) | ~1M combinations → ~1s |
| **QAOA** (simulated, p=2) | O(2^(N·m) × p × iters) | >10 min (exponential!) |

**Speedup QA vs SA**: ~15ms / 0.02ms = **×750 faster** (anneal only)
**עם communication overhead**: ~15ms / 12ms = **×1.25** (bottleneck = network)

→ ב-N=10 ה-speedup מתון בגלל communication. ב-N גדול (SA לוקח שניות) ה-speedup גדל ל-**×1000+**.

---

## עוד שנה: Advantage2 Performance Update (2026-2027)

### שיפורים צפויים
- **Cyclic Annealing**: warm-start מפתרון קודם → anneal time יורד ל-**~5μs**
- **Novel annealing protocols**: reverse + forward anneals
- **אותם 4,400 qubits**, protocol טוב יותר

### מה ישתנה

| מאפיין | היום | 2027 |
|---------|------|------|
| Anneal time | 20μs | ~5μs (warm-start) |
| Solution quality | good | **better** (cyclic refines) |
| Max N (fully connected, m=4) | 17 | 17 (same hardware) |
| Max N (sparse, d=4) | ~200 | ~200 (same hardware) |
| Regret | low | **lower** (warm-start = less exploration waste) |

**Impact on QA-MAB**: Cyclic annealing מושלם למודל שלנו — בstep t+1 מתחילים מהפתרון של step t. הQUBO כמעט לא משתנה בין steps (τ ו-I_hat משתנים בהדרגה). Warm-start = **convergence מהיר יותר**.

---

## 3-5 שנים קדימה (2028-2031)

### תחזיות חומרה

| דור | שנה (est.) | Qubits | Connectivity | Max K_n |
|-----|-----------|--------|-------------|---------|
| Advantage2 | 2025 | 4,400 | 20-way | ~68 |
| Advantage3 (est.) | 2028 | ~10,000 | ~30-way | ~170 |
| Advantage4 (est.) | 2030 | ~20,000+ | ~40-way | ~280 |

*הערכות מבוססות על מגמת ×2-3 qubits כל 2-3 שנים ושיפור connectivity*

### מה נוכל לפתור

#### Fully Connected

| דור | Max N (m=4) | Max N (m=8) |
|-----|------------|------------|
| Advantage2 (2025) | **17** | **8** |
| Advantage3 (2028) | **~42** | **~21** |
| Advantage4 (2030) | **~70** | **~35** |

#### Sparse (degree d=4)

| דור | Max N (m=4) | Max N (m=8) |
|-----|------------|------------|
| Advantage2 (2025) | **~200** | **~100** |
| Advantage3 (2028) | **~500** | **~250** |
| Advantage4 (2030) | **~1,000+** | **~500** |

### סיבוכיות זמן — היום vs עתיד

```
SA Solver:
  T(N,m) = O(restarts × iters × N × m)
  N=10:  ~15ms
  N=50:  ~200ms  
  N=100: ~800ms
  N=500: ~20s
  (גדל ליניארית עם N, אבל איכות הפתרון יורדת!)

D-Wave QA (היום):
  T(N,m) = O(1) anneal + O(N·m) embedding
  N=10:  ~12ms (anneal 20μs + communication 10ms)
  N=50:  ~12ms (same!)
  N=100: ~15ms (embedding מורכב יותר)
  N=500: ~20ms (sparse embedding)
  (כמעט קבוע! bottleneck = communication, לא חישוב)

D-Wave QA (2028, est.):
  T(N,m) = O(1) anneal + O(1) on-chip embedding
  N=10:  ~5ms
  N=50:  ~5ms
  N=100: ~5ms
  N=500: ~8ms
```

### טבלת Speedup

| N | m | SA time | QA time (2025) | QA time (2028 est.) | Speedup 2025 | Speedup 2028 |
|---|---|---------|---------------|-------------------|-------------|-------------|
| 10 | 4 | 15ms | 12ms | 5ms | ×1.3 | ×3 |
| 50 | 4 | 200ms | 12ms | 5ms | **×17** | **×40** |
| 100 | 4 | 800ms | 15ms | 5ms | **×53** | **×160** |
| 500 | 4 | 20s | 20ms | 8ms | **×1,000** | **×2,500** |

**חשוב:** ה-speedup הוא רק בזמן. היתרון **האמיתי** של QA הוא **איכות הפתרון** — quantum tunneling מוצא global optimum שSA מחמיץ. זה מתורגם ל-regret נמוך יותר, לא רק חישוב מהיר יותר.

---

## סיכום: למה QA-MAB רלוונטי

1. **היום** — N≤17 fully connected, N≤200 sparse. מספיק להוכחת קונספט
2. **2028** — N≤42 fully, N≤500 sparse. רשתות אמיתיות קטנות-בינוניות
3. **2030+** — N≤70 fully, N≤1000 sparse. רשתות אמיתיות בסקלה
4. **Speedup גדל עם N** — SA מתדרדר ליניארית, QA נשאר קבוע
5. **איכות עולה עם חומרה** — coherence + connectivity = tunneling טוב יותר = regret נמוך יותר

### השורה התחתונה לתזה
> QA-MAB הוא framework שמתאים **כבר היום** לרשתות קטנות-בינוניות, ו**ישתפר אוטומטית** עם כל דור חדש של חומרה קוונטית — ללא שינוי באלגוריתם.

---

## Links
- [[qa-mab-simulation-results]] — תוצאות סימולציה
- [[DIAMOND-paper-notes]] — המאמר המקורי

## Tags
#thesis #quantum #hardware #d-wave #scaling #future-work
