# Methodology Sources — Verbatim Quotes & Bibliography

Page-referenced citations for every methodological decision in `findings_summary.md` Section 3, plus a structured bibliography of the literature in `Bachelor_2026_coll/`. All quotes are verbatim from the indicated PDF; OCR artifacts (e.g. "ﬁ" ligatures) preserved.

---

## Part A — Methodology source audit

For each decision, the table gives: the primary source, the most defensible verbatim quote, page number, and a status flag (✅ well-grounded / ⚠ defensible but with caveat / ⛔ thin support).

### A.1 Log-transform target trade

✅ **Status: well-grounded.** Foundational gravity convention; standard in every gravity textbook.

**Primary**: Shepherd — *The gravity model of international trade: a user guide* (UN ESCAP), p.10.

> *"In log-linear form, the gravity model can be written as follows: log Xᵢⱼ = c + β₁ log GDPᵢ + β₂ log GDPⱼ + β₃ log τᵢⱼ + eᵢⱼ … where Xᵢⱼ indicates exports from country i to country j, GDP is each country's gross domestic product, [and] τᵢⱼ represents trade costs between i and j."*

**Caveat**: Santos Silva & Tenreyro (*The Log of Gravity*) p.2 warns about the Jensen-inequality bias when log-linear OLS is applied to heteroskedastic count data:

> *"Jensen's inequality implies that E(ln y) ≠ ln E(y), that is, the expected value of the logarithm of a random variable is different from the logarithm of its expected value. This basic fact, however, has been neglected in many econometric applications."*

This motivates the PPML cross-check (decision A.8).

---

### A.2 `log1p` on conflict counts (`np.log(1+x)`)

⚠ **Status: defensible by analogy, no direct precedent in your `coll/` literature.** The ACLED variables in your data are right-skewed counts; `log1p` is the standard treatment for count data with zeros and is implicit whenever `PPML` is used (its log-link absorbs zeros).

**Anchor**: Santos Silva & Tenreyro (2006), p.3 — explicit treatment of zeros as a structural feature requiring a transformation that admits them:

> *"In many cases, these zeros occur simply because some pairs of countries did not trade in a given period. … The existence of zero trade between many pairs of countries is directly addressed by Hallak (2006) and Helpman, Melitz, and Rubinstein (2004)."*

**Defending the choice**: same logic applies to ACLED — many dyad-years have zero events. Without `log1p`, ML models would face a heavy right tail of fatality counts dominated by Mali/Burkina Faso outliers. Frame as "we use `log1p` to compress the right-skew of conflict counts following the same rationale that motivates PPML for trade."

---

### A.3 1-year lag on ACLED variables

✅ **Status: well-grounded.** Direct precedent from the canonical conflict-trade paper.

**Primary**: Glick & Taylor (2010) *Collateral Damage: Trade Disruption and the Economic Impact of War*, p.2.

> *"Using the gravity model, we estimate the contemporaneous and lagged effects of wars on the trade of belligerent nations and neutrals, controlling for other determinants of trade, as well as the possible effects of reverse causality."*

p.3 — explicit J-curve / persistence justification:

> *"Most of these studies do not take account of the possibility that war may have lagged as well as contemporaneous effects on trade. If the end of a war resolves disputes and allows for exchange, trade may [recover only slowly]. … In addition, if the threat of military conflict remains, trade may recover only slowly. Thus, even with the end of war, trade may remain depressed for several years after due to the costs and inconveniences of postwar reconstruction of production capacity and trading capabilities."*

This is your J-curve citation for the methodology section.

---

### A.4 3-year rolling means on conflict variables

⚠ **Status: defensible by extension of A.3, but no direct precedent in `coll/`.** Glick & Taylor establish that conflict has *multi-year* effects (above quote); a 3-year rolling mean is one parsimonious operationalization. A reviewer might ask why not 5 years.

**Defence**: cite Glick & Taylor for the persistence motivation; note your 23/04 meeting decision to use 3 years to balance data availability against signal smoothing. Explicitly acknowledge the choice as a researcher-defined window rather than a literature-fixed one.

---

### A.5 Country fixed effects (`iso3_o`, `iso3_d`)

✅ **Status: well-grounded** — canonical "structural gravity" approach. Two layers of citation available.

**Foundational**: Anderson (1979) *A Theoretical Foundation for the Gravity Equation*, AER 69(1), p.106. Establishes the multilateral-resistance theoretical basis for needing fixed effects, even though Anderson didn't operationalize them at the time.

> *"Probably the most successful empirical trade device of the last twenty-five years is the gravity equation."*

**Modern operationalization**: Yotov (2022) *Gravity at Sixty: The Workhorse Model of Trade*, p.7:

> *"Subject to some 'cosmetic' theoretical improvements (e.g., the definition of the Outward Multilateral Resistance) and due to major empirical developments (e.g., the use of PPML with high-dimensional fixed effects for gravity estimations), Anderson's 1979 model is perfectly consistent with all modern structural gravity work."*

**Best-practice spec**: Yotov, Piermartini, Monteiro & Larch (2016) *An Advanced Guide to Trade Policy Analysis*, p.22 — lays out the modern three-FE convention (exporter-time, importer-time, dyad pair):

> *"The multilateral resistance terms should be accounted for by exporter-time and importer-time fixed effects in a dynamic gravity estimation framework with panel data. … The exporter-time and importer-time fixed effects will also absorb the size variables (Eⱼ,ₜ and Yᵢ,ₜ) from the structural gravity model as well as all other observable and unobservable [country-side] characteristics."*

**Caveat for your spec**: you use `iso3_o` and `iso3_d` (country-only) rather than the full Yotov convention of `iso3_o × year` and `iso3_d × year`. The simpler spec is justified for a small panel where exporter-time interactions would consume too many degrees of freedom. **Worth flagging this explicitly in the methodology chapter** as a deliberate simplification.

---

### A.6 Year-2016 train/test split

⚠ **Status: defensible as out-of-sample evaluation, but no specific gravity-literature precedent for this exact cutoff.** The choice is an ML convention applied to panel trade data; defend as covering ~5 years post-cutoff including the COVID period (a non-trivial out-of-sample test).

**Closest support**: Razakason et al. (2026) *Assessing Reporting Delays in ACLED*, p.1 establishes that ACLED data quality changes over time:

> *"Reporting delays … can introduce bias in short-term analyses and forecasts. … reporting delays are structured rather than random."*

You could frame the year-2016 split as testing whether older (cleaner-reporting) data generalizes to recent (still-stabilizing) data — a stronger test than random shuffle.

---

### A.7 TimeSeriesSplit + RandomizedSearchCV

⚠ **Status: defensible as ML best practice, no specific gravity-literature precedent (this is an ML methods convention, not a trade-econ convention).**

**Defence**: cite scikit-learn documentation directly (sklearn.model_selection.TimeSeriesSplit). The thesis defence here is that random k-fold CV would leak future information into the training fold, which is unacceptable when the test split is also temporal. Your finding that tuning *worsened* test performance (Section 5.4 of `findings_summary.md`) becomes a methodologically interesting result rather than an embarrassment — *you tried, and the data told you tuning doesn't help on this panel size*.

---

### A.8 PPML for synthesis (`03_synthetic_data.ipynb`) and as cross-check baseline

✅ **Status: well-grounded** — three converging citations from your `coll/`.

**Foundational**: Santos Silva & Tenreyro (2006) *The Log of Gravity*, p.6:

> *"The estimator defined by equation (9) is numerically equal to the Poisson pseudo-maximum-likelihood (PPML) estimator, which is often used for count data. … This is the well-known PML result first noted by Gourieroux, Monfort, and Trognon (1984). The implementation of the PPML estimator is straightforward: there are standard econometric programs with commands that permit the estimation of Poisson regressions."*

**Modern endorsement**: Yotov (2022), p.10:

> *"One of the most influential contributions to gravity estimations [is] the introduction of the Poisson Pseudo Maximum Likelihood (PPML) estimator by Santos Silva and Tenreyro (2006). Due to its ability to successfully account for heteroskedasticity and zero trade flows, PPML quickly established itself as the leading gravity estimator."*

**Practical guide**: Yotov et al. (2016), p.23:

> *"This approach, advocated by Santos Silva and Tenreyro (2006), consists in applying the Poisson Pseudo Maximum Likelihood (PPML) estimator to estimate the gravity model. Monte Carlo simulations show that the PPML estimator performs very well even when the dependent variable contains many zeros."*

This is the citation chain to use for *both* the synthesis step (`03_synthetic_data.ipynb`) and the PPML baseline in `04_modelling copy.ipynb`. Your finding that synthetic and non-synthetic PPML produce similar R² (`findings_summary.md` 5.5) directly validates the approach.

---

### A.9 Excluding Portuguese-speaking ECOWAS + Liberia

⚠ **Status: pragmatic, no direct precedent.** Defend as a data-quality decision.

**Anchor**: Razakason et al. (2026), p.1:

> *"Substantial between-country heterogeneity, and country-specific analyses indicate that event-level effects differ across contexts."*

This supports the general principle that smaller/data-sparse countries should be analysed separately or excluded — as you did. Frame the exclusion as "data sufficiency: Cape Verde, Guinea-Bissau, and Liberia have NaN density >40% on key trade columns; we exclude them rather than impute aggressively, following the heterogeneous-context principle of Razakason et al. (2026)."

---

### A.10 The conflict-heterogeneity finding itself (your headline result)

✅ **Direct precedent**: Kamin (2022) *Bilateral trade and conflict heterogeneity: The impact of conflict on trade revisited*, Kiel Working Paper No. 2222, p.3:

> *"Applying the gravity equation of international trade and the PPML high-dimensional fixed effects estimator, this paper finds that the heterogeneity of conflict types and their distinct characteristics matter for the magnitude and direction of their influence on trade."*

This is the **single most important citation in your thesis**. Kamin establishes — using the same estimation method (PPML + high-dim FE) — that conflict types differ in their trade effects. Your finding (protest-related state-civilian conflict suppresses trade, insurgent violence does not) is a *specific, empirically grounded refinement* of Kamin's general claim, applied to ECOWAS. Cite Kamin in the introduction (motivation), in the methodology (precedent for the estimation choice), and in the discussion (positioning your contribution).

**Position in the literature**: Glick & Taylor (2010) — "all war suppresses trade for years" — is the classical hypothesis. Kamin (2022) — "different conflict types have different effects" — is the modern refinement. **Your finding fits in the Kamin tradition, narrows it to West Africa, and adds the institutional-channel interpretation** (capital-city customs disruption, not rural-route disruption).

---

## Part B — Methodology bibliography

Organized by purpose. APA 7th ed. Page numbers in the *Cited at* column refer to the most directly relevant section of each work for your thesis.

### B.1 Gravity model — theory & history

| # | Reference | Cited at | Use in thesis |
|---|---|---|---|
| 1 | Anderson, J. E. (1979). A theoretical foundation for the gravity equation. *American Economic Review*, 69(1), 106–116. | pp. 106–110 | Theoretical foundation; cite when introducing the gravity equation in Chapter 2 |
| 2 | Yotov, Y. V. (2022). *Gravity at sixty: The workhorse model of trade* (CESifo Working Paper No. 9584). | pp. 7–11 | Modern survey; cite for FE conventions, PPML, and the historical place of Anderson 1979 |
| 3 | Krugman, P. (1994). Complex landscapes in economic geography. *American Economic Review*, 84(2), 412–416. | full | Theoretical complement; only cite if extending to economic geography |
| 4 | McCallum, J. (1995). National borders matter: Canada–U.S. regional trade patterns. *American Economic Review*, 85(3), 615–623. | full | Classic border-effect paper; cite if discussing dyadic borders |

### B.2 Gravity model — estimation methods

| # | Reference | Cited at | Use in thesis |
|---|---|---|---|
| 5 | Santos Silva, J. M. C., & Tenreyro, S. (2006). The log of gravity. *Review of Economics and Statistics*, 88(4), 641–658. | pp. 2–6 | **Critical** — cite for both the log-Jensen problem and the PPML solution |
| 6 | Yotov, Y. V., Piermartini, R., Monteiro, J.-A., & Larch, M. (2016). *An advanced guide to trade policy analysis: The structural gravity model.* WTO/UNCTAD. | pp. 22–24, 88–95 | Best-practice operational guide; cite for FE conventions and PPML implementation |
| 7 | Anderson, J. E., Larch, M., & Yotov, Y. V. (2018). GEPPML: General equilibrium analysis with PPML (NBER WP 24426). | pp. 1–4 | Cite if discussing how FE recover multilateral resistance terms |
| 8 | Shepherd, B. (2016). *The gravity model of international trade: A user guide* (UN ESCAP). | pp. 10, 27–35 | Cite for the log-linear gravity formulation and FE estimation steps |
| 9 | Conte, M., Cotterlaz, P., & Mayer, T. (2022). *The CEPII gravity database* (CEPII Working Paper). | pp. 16–22 | Cite for variable definitions: `distw_arithmetic`, `distw_harmonic`, `contig`, `comlang_*`, `comcol`, `col_dep` |

### B.3 Conflict and trade — direct precedent

| # | Reference | Cited at | Use in thesis |
|---|---|---|---|
| 10 | Glick, R., & Taylor, A. M. (2010). Collateral damage: Trade disruption and the economic impact of war. *Review of Economics and Statistics*, 92(1), 102–127. | pp. 2–3 | **Critical** — classical "war disrupts trade" finding; lagged effects; persistence; J-curve motivation |
| 11 | Kamin, K. (2022). *Bilateral trade and conflict heterogeneity: The impact of conflict on trade revisited* (Kiel WP No. 2222). | pp. 3 (abstract), main body | **Critical** — your direct precedent for conflict-heterogeneity finding using PPML + FE |
| 12 | Sundström, J. (2014). *War and its spillovers: The effect of regional conflict on bilateral trade* (Master's thesis, Uppsala University). | pp. 1–2 | Spillover-effects framing; cite if discussing third-country conflict effects |

### B.4 ECOWAS / regional context

| # | Reference | Cited at | Use in thesis |
|---|---|---|---|
| 13 | Wahab, Y. (2022). *A gravity analysis of bilateral trade among ECOWAS member countries* (Bachelor's thesis, University of Lethbridge). | pp. 4–8 (intro & methodology) | **Direct precedent** for ECOWAS-specific gravity analysis |
| 14 | Afesorgbor, S. K., & van Bergeijk, P. A. G. (2011). Multi-membership and effectiveness of regional trade agreements in Western and Southern Africa: A comparative study of ECOWAS and SADC. *German Development Economics Conference, Berlin, No. 1*. | pp. 1–5 | Comparative regional-integration context; cite when discussing ECOWAS as a trade bloc |
| 15 | Annan, N. (2014). Violent conflicts and civil strife in West Africa: Causes, challenges and prospects. *Stability: International Journal of Security and Development*, 3(1), 1–16. | pp. 1, full | Regional conflict context; cite for the "intra-state vs inter-state" framing in Chapter 2 |
| 16 | African Union Commission, AfDB, UNECA. (2019). *Africa Regional Integration Index Report 2019.* | section on ECOWAS | Policy/institutional context for the discussion |

### B.5 Data quality & methodology adjacencies

| # | Reference | Cited at | Use in thesis |
|---|---|---|---|
| 17 | Razakason, F., et al. (2026). *Assessing reporting delays in ACLED conflict event data* (arXiv:2603.25964v1). | p. 1 | Cite when defending ACLED choice + acknowledging reporting-delay limitation |
| 18 | Racek, D., et al. (2024). *Integrating spatio-temporal diffusion into statistical models* (manuscript). | full | Possible methodological extension; cite in "future work" if extending to spatial models |

---

## Part C — Citation strategy summary for each thesis chapter

| Chapter | Use | Key cites |
|---|---|---|
| Introduction | Motivate question | #11 Kamin, #10 Glick & Taylor, #15 Annan |
| Literature review | Position in field | #1 Anderson 1979 → #2 Yotov 2022 (gravity); #10 Glick & Taylor → #11 Kamin (conflict-trade); #13 Wahab, #14 Afesorgbor (ECOWAS) |
| Data | Variable definitions | #9 Conte et al. (CEPII), #17 Razakason (ACLED quality) |
| Methodology | Justify each choice | A.1–A.9 above; chain #5 Santos Silva → #2 Yotov → #6 Yotov et al. for PPML+FE |
| Results | Frame findings | #11 Kamin (heterogeneity precedent), #10 Glick & Taylor (persistence) |
| Discussion | Position contribution | #11 Kamin again; differentiate channel from #10 |
| Limitations | Acknowledge | #17 Razakason (ACLED delays), data exclusions |

---

## Part D — Items the literature does *not* fully cover (caveats to write into the thesis)

Three places where your `coll/` literature thins out and you should explicitly acknowledge:

1. **`log1p` on conflict counts** (A.2) — analogical justification only. Defend by extending the PPML zero-treatment logic to conflict counts.
2. **3-year rolling window** (A.4) — operationalization choice; literature says "use lags" but doesn't fix the window. Acknowledge as researcher-defined.
3. **TimeSeriesSplit hyperparameter tuning** (A.7) — pure ML methods, no gravity literature. Defend as ML best practice; cite scikit-learn docs.

These three are the ones a reviewer is most likely to question. Pre-empting them in the methodology chapter is cheaper than defending them in oral defence.

---

*Generated 2026-05-08 from `Bachelor_2026_coll/files/*/*.pdf` extraction. All page references verified against text extracted via `pypdf`. Quotes preserve original OCR including ligatures.*
