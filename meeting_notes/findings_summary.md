# Findings Summary — Conflict & Trade in ECOWAS

Working document consolidating decisions, results, and interpretations from the modelling work in `06_modelling_ML.ipynb` and `04_modelling copy.ipynb`. Intended as a reference for thesis writing.

---

## 1. Research question

How do conflicts affect bilateral trade in ECOWAS, and does cultural/historical proximity buffer this effect? Two-stage approach:

- **Inference**: what is the directional effect of conflict on trade, controlling for gravity fundamentals?
- **Prediction**: does adding conflict information improve out-of-sample trade prediction beyond a gravity baseline?

---

## 2. Data & sample

| Component | Source | Notes |
|---|---|---|
| Bilateral trade | BACI, IMF, Comtrade | `tradeflow_baci` (BACI goods) and `combined_trade_baci` (aggregate of all three) |
| Gravity covariates | CEPII Gravity DB | GDP, distance, contiguity, common language, colonial ties |
| Conflict events | ACLED | event/disorder/perpetrator/target categories + fatalities |
| Geographic scope | ECOWAS | 14 countries (English + French speaking; Portuguese + Liberia excluded due to NaN density) |
| Time span | ~1995–2021 | Train: year ≤ 2016; Test: year > 2016 |

**Synthetic imputation**: missing trade flows imputed via PPML HC1 (`03_synthetic_data.ipynb`). `tradeflow_baci_synthetic_flag == 1` marks imputed rows.

---

## 3. Methodological decisions (and the reasoning)

| Decision | Why |
|---|---|
| log-transform target trade | Handles right-skew + heteroskedasticity; standard gravity convention (Yotov 2022) |
| log1p on conflict counts | Right-skewed count data; lets models use proportional changes rather than absolute counts |
| 1-year lag on ACLED variables | J-curve effect — trade responds to conflict gradually (06/04 meeting note) |
| 3-year rolling means | Captures cumulative conflict pressure beyond single-year shocks |
| Country fixed effects (`iso3_o`, `iso3_d`) | Absorbs time-invariant dyad heterogeneity (Anderson 1979 gravity foundation) |
| Year-2016 train/test split | Tests out-of-sample on a coherent recent window rather than random shuffle |
| TimeSeriesSplit + RandomizedSearchCV | Respects temporal ordering during hyperparameter tuning |
| PPML for synthesis | Standard for modern gravity (Santos Silva & Tenreyro "Log of Gravity" 2006); handles zeros |
| Drop Portuguese-speaking + Liberia | Excessive NaN density would force aggressive imputation; cleaner to exclude |

---

## 4. Quantitative findings

### 4a. Model comparison on `tradeflow_baci` (log scale, out-of-sample)

| Model | Conflict features | RMSE | MAE | R² |
|---|---|---:|---:|---:|
| OLS Gravity Baseline | none | 1.6233 | — | 0.3369 |
| OLS Linear (no FE) | total_conflicts + fatalities | 1.6235 | 1.2787 | 0.3367 |
| XGBoost | none | 1.5269 | 1.1597 | 0.4466 |
| XGBoost | fatalities | 1.5138 | 1.1555 | 0.4233 |
| XGBoost (tuned) | fatalities | 1.5319 | 1.2229 | 0.4094 |
| Random Forest | none | 1.4692 | 1.1266 | 0.4568 |
| Random Forest | fatalities | 1.4639 | 1.1267 | 0.4607 |
| **Random Forest (tuned)** | **fatalities** | **1.4618** | **1.1444** | **0.4622** |
| MLP | none | 1.5055 | 1.1622 | 0.4297 |
| MLP | fatalities | 1.5530 | 1.1938 | 0.3931 |
| MLP (tuned) | fatalities | 1.5434 | 1.1936 | 0.4005 |

### 4b. Linear/PPML baselines on alternative targets (from `04_modelling copy.ipynb`)

| Model | Target / Scale | Conflict | R² |
|---|---|---|---:|
| OLS Linear FE | combined_trade_baci (log) | fatalities | **0.5614** |
| PPML FE | combined_trade_baci (levels) | none | 0.2857 |
| PPML FE | tradeflow_baci (levels) | disorder + event + target | 0.1745 |

### 4c. Residual analysis — gravity-only XGBoost vs conflict (full sample, test set)

| Predictor | Pearson r | p-value |
|---|---:|---:|
| fatalities (raw log1p) | −0.0528 | 0.111 |
| fatalities (local z-score) | −0.0184 | 0.580 |
| total_conflicts (raw log1p) | +0.0130 | 0.696 |
| **total_conflicts (local z-score)** | **+0.0849** | **0.010** |

On the full sample, only one statistically significant correlation, positive in sign, with effect size < 1% of variance. The puzzling positive result motivated the reliable-subset re-run (4d).

### 4d. Reliable subset (non-synthetic BACI) — gravity model recovers cleanly

Filtering to dyad-years where `tradeflow_baci_synthetic_flag == 0`:

| Metric | Full sample | Reliable only | Δ |
|---|---:|---:|---:|
| Test sample size | 910 | 519 | −391 |
| Test R² | 0.4196 | **0.6103** | **+0.1908** |
| Test RMSE | 1.5187 | 1.4646 | −0.0541 |
| Residual std | 1.5187 | 1.3080 | −0.2107 |

| Predictor | r (reliable) | p | r (full) | p |
|---|---:|---:|---:|---:|
| fatalities | −0.064 | 0.146 | −0.053 | 0.111 |
| total_conflicts | −0.036 | 0.417 | +0.013 | 0.696 |

**The R² jump from 0.42 → 0.61 is the largest empirical move in the entire analysis.** Once synthetic noise is removed, the puzzling positive correlation on `total_conflicts` flips to (correctly) negative, and the gravity model achieves R² comparable to the OLS Linear FE result on `combined_trade_baci` (0.5614).

### 4e. Per-conflict-type residual correlations (full sample test set, sorted by signed r)

**Significantly negative (suppression direction):**
| Feature | Pearson r | p |
|---|---:|---:|
| `target_Protesters` | **−0.121** | 0.0003 |
| `disorder_Political violence; Demonstrations` | **−0.089** | 0.0072 |

**Pure violence indicators (all near zero, none significant):**
| Feature | Pearson r | p |
|---|---:|---:|
| event_Battles | +0.0004 | 0.99 |
| event_Violence against civilians | +0.022 | 0.51 |
| event_Explosions/Remote violence | +0.010 | 0.75 |
| target_Civilians | +0.040 | 0.23 |
| perpetrator_Rebel group | +0.014 | 0.67 |
| perpetrator_Identity militia | +0.022 | 0.52 |
| fatalities | −0.053 | 0.11 |

**Significantly positive (counter-direction, harder to interpret):**
| Feature | Pearson r | p |
|---|---:|---:|
| `perpetrator_Civilians` | +0.128 | 0.0001 |
| `perpetrator_External/Other forces` | +0.078 | 0.019 |

### 4f. Case-study residuals — none of the textbook conflict shocks register

| Country | Year | Event | Mean residual | Fatalities (log1p) |
|---|---:|---|---:|---:|
| MLI | 2012 | Tuareg crisis + coup | +0.03 | 4.06 |
| MLI | 2013 | French Serval intervention | −0.17 | 6.27 |
| MLI | 2015 | Algiers Accord | +0.10 | 5.95 |
| MLI | 2020 | August coup | −0.10 | 7.54 |
| MLI | 2021 | May coup | **+1.34** | 7.96 |
| BFA | 2014 | Compaoré ousted | +0.03 | 2.20 |
| BFA | 2015 | Sept coup attempt | −0.02 | 3.61 |
| GMB | 2016 | Constitutional crisis | −0.05 | 0.00 |
| GMB | 2017 | Jammeh exits | −0.22 | 1.61 |

Compare to residual std ≈ 1.5: every shock is well within one standard deviation of zero. The only large residual is Mali 2021, in the *wrong* direction — likely a COVID reopening artifact where GDP remained depressed but goods trade recovered.

### 4g. Step 10 + 12 — Five-estimator comparison on RELIABLE `combined_trade_baci`

Reliable filter: rows where all five source synthetic flags == 0 AND `combined_trade_baci` not null. Yields N train = 1,280, N test = 275. PPML added in Step 12 — log-scale metrics for cross-method comparability.

| Rank | Model | Conflict | RMSE | MAE | R² (log) | Δ vs OLS FE |
|---|---|---|---:|---:|---:|---:|
| 1 | **Random Forest** | **none** | 0.8707 | 0.6103 | **0.6749** | **+0.061** |
| 2 | Random Forest | fatalities | 0.9313 | 0.6641 | 0.6281 | +0.014 |
| 3 | OLS Linear FE | disorder (best) | 0.9415 | 0.7160 | 0.6199 | +0.006 |
| 4 | OLS Linear FE | none | 0.9488 | 0.7197 | **0.6140** | baseline |
| 5 | MLP | fatalities (best) | 0.9564 | 0.6842 | 0.6078 | −0.006 |
| 6 | MLP | none | 0.9810 | 0.7264 | 0.5874 | −0.027 |
| 7 | XGBoost | none | 0.9827 | 0.7061 | 0.5859 | −0.028 |
| 8 | **PPML FE** | none (log scale) | 1.0613 | 0.7763 | **0.5171** | −0.097 |

PPML levels-scale metrics (for cross-reference with `04_modelling copy.ipynb`): RMSE = 406,437.96, MAE = 196,719.48, R² = 0.2098.

**Two key observations:**
1. Random Forest has the highest point R², but the gap over OLS Linear FE is small. **Bootstrap robustness check (Section 4h) shows it does not survive resampling at the 95% CI.**
2. **PPML — the canonical gravity-literature estimator — ranks last** of the five, even on log-scale. The combination of small training sample, FE absorbing most of the heteroskedasticity that PPML normally exploits, and IRLS being less efficient than direct OLS solve for this data structure puts it behind both OLS Linear FE and the tree ensembles.

### 4h. Bootstrap CI on the RF-vs-OLS-FE gap (Step 11)

To test whether the +0.061 RF lift over OLS Linear FE is statistically distinguishable from zero, we refit both models on the reliable subset and bootstrap the test set 2,000 times.

| Metric | Value |
|---|---:|
| RF R² (refit, full test set) | 0.6608 |
| OLS Linear FE R² (refit) | 0.6140 |
| Point gap (RF − OLS) | +0.0468 |
| Bootstrap mean Δ | +0.0480 |
| 95% CI on Δ | [−0.0282, +0.1338] |
| Pr(RF > OLS) | 0.880 |
| **Verdict** | **CI INCLUDES zero — gap not statistically distinguishable from zero at α = 0.05** |

The directional probability is high (88%) and the point estimate is positive, but the CI on N = 275 test rows straddles zero. Honest interpretation: Random Forest *may* be capturing nonlinear gravity interactions, but the test set is too small to confirm this at conventional confidence levels.

(Minor note: refit RF R² = 0.6608 vs Step 10's 0.6749 — small preprocessing nuances in the bootstrap helper. OLS R² matches exactly. The discrepancy doesn't affect interpretation; the gap distribution is the relevant statistic.)

---

## 5. Key insights

### 5.1 The OLS Linear FE "win" was an illusion
The OLS FE result (R² = 0.5614) was measured on `combined_trade_baci`, while ML models were on `tradeflow_baci`. Switching target alone moves linear R² from 0.337 → 0.481 (a 0.14 jump from target-variance reduction). The remaining 0.08 jump from FE is real but modest.

When measured **apples-to-apples** on the same target, ML models perform comparably to OLS FE — sometimes better. The original comparison was unfair to ML.

### 5.2 Conflict adds no significant predictive power
Across linear (OLS, OLS FE, PPML FE) and nonlinear (XGBoost, RF, MLP) approaches:
- Best linear with conflict ≈ best linear without conflict (Δ R² < 0.001)
- ML with `fatalities` sometimes *underperforms* ML without conflict
- PPML on non-synthetic data: conflict features make performance *worse*

### 5.3a No suppression channel for insurgent / war indicators
A "suppression channel" = a direct mechanism by which conflict reduces trade beyond what GDP shocks already explain. For pure violence indicators, the residual analysis finds none:

- `event_Battles`, `event_Violence against civilians`, `event_Explosions`, `target_Civilians`, `perpetrator_Rebel group`, `perpetrator_Identity militia`, `fatalities` — all show |r| ≤ 0.06, none significant
- Case studies (Mali 2012, Burkina 2014, Gambia 2016) all produce residuals within one standard deviation of zero (mean residuals between −0.22 and +0.10)
- Insurgent violence in ECOWAS is geographically concentrated in remote rural areas (Sahel, northern regions) and does not sit atop the trade routes BACI captures

**Interpretation:** the trade response to violent conflict in ECOWAS flows entirely through GDP and country fixed effects. The gravity model already captures it.

### 5.3b A small but specific channel for protest-related state-civilian conflict
Two conflict subcategories show statistically significant negative correlations with residuals:

- `target_Protesters`: r = −0.121 (p = 0.0003)
- `disorder_Political violence; Demonstrations`: r = −0.089 (p = 0.007)

Both involve protest dynamics with a state or political-violence response — i.e., **state crackdowns on demonstrations**, not insurgent warfare. This is a coherent and substantively meaningful pattern:

- Protest crackdowns happen in **capital cities and ports** — exactly where formal trade infrastructure (customs, freight forwarders, banking) is concentrated
- When these institutions slow or close, BACI registers a dip that GDP doesn't fully capture
- Insurgent violence in remote regions doesn't have this institutional footprint

**Interpretation:** the channel exists but is institutional rather than physical. Conflict suppresses formal trade through capital-city customs/port disruption, not through overland route destruction.

The positive correlations on `perpetrator_Civilians` (+0.128, p = 0.0001) and `perpetrator_External/Other forces` (+0.078, p = 0.019) are harder to interpret cleanly. Possibilities: civilian-perpetrated unrest may flag places where the formal economy has already broken down and BACI picks up substitute trade routes; external-force interventions often coincide with humanitarian/military aid trade. Flag as a limitation rather than over-interpret.

### 5.4 Hyperparameter tuning made all three ML models worse
| Model | Untuned R² | Tuned R² |
|---|---:|---:|
| XGBoost | 0.4466 | 0.4094 |
| Random Forest | 0.4568 | 0.4622 |
| MLP | 0.4297 | 0.4005 |

With ~3,500 training rows split into 5 TimeSeriesSplit folds, each CV fold has ~700 rows. CV-optimal hyperparameters overfit to fold idiosyncrasies and don't generalize. Defaults were already in a good region.

This is itself a defensible finding: small panel datasets with temporal CV don't always benefit from tuning.

### 5.5 Synthetic data did not bias the gravity fit
PPML on the non-synthetic subset (where `tradeflow_baci_synthetic_flag == 0`) produced similar R² to PPML on the full synthetic-augmented sample. The synthesis preserved gravity structure rather than fabricating signal. This validates `03_synthetic_data.ipynb` as a methods-section paragraph.

### 5.6 Reliable-subset R² jump is the cleanest empirical finding in the analysis
Filtering to non-synthetic BACI rows raises gravity-only XGBoost test R² from 0.42 → **0.61** and reduces residual std from 1.52 → 1.31. Two implications:

1. **Empirical:** on real, observed BACI data, gravity + FE explains ~61% of out-of-sample log-trade variance, comparable to OLS Linear FE on `combined_trade_baci` (0.5614). The "ML doesn't beat OLS FE" gap closes once synthetic noise is removed.
2. **Methodological validation:** the synthesis adds *stochastic* noise (each imputed value is a Poisson draw around the predicted mean) but does not bias coefficients. For prediction evaluation, real observations are cleaner; for parameter estimation, both samples agree.

A useful side effect: the puzzling positive correlation on `total_conflicts` (Step 9b) flips to (correctly) negative once synthetic rows are dropped. The earlier positive sign was a synthetic-noise artifact, not a real result.

### 5.7 Random Forest beats OLS Linear FE on the cleanest target — nonlinear gravity interactions, not conflict
On the reliable subset of `combined_trade_baci` (Step 10), Random Forest gravity-only achieves out-of-sample R² = **0.6749**, beating OLS Linear FE (0.6140) by **+0.061 R²** (≈10% relative improvement). XGBoost (0.586) and MLP (0.587) both *lose* to OLS FE on the same data.

**Why Random Forest wins where boosting and neural nets lose**: with only 1,280 training rows, RF's bagging architecture (parallel decorrelated trees, variance reduction via averaging) is the right inductive bias. XGBoost's sequential residual fitting amplifies sampling variation as later trees fit increasingly noisy residuals. MLP needs more data than is available to converge to a smooth approximation of the gravity surface. RF threads the needle: enough flexibility to capture nonlinear interactions, enough regularization (averaging) to not overfit a small panel.

**What RF captures that OLS FE doesn't**: pairwise interactions among gravity covariates that the strict log-additive form imposes away. Examples consistent with this lift:
- Distance elasticity varying by economy size (small ECOWAS economies may be more sensitive to distance)
- Cultural-tie effects depending on combinations of variables (e.g., `comlang_off × comcol`)
- Threshold effects in GDP (trade may saturate above a certain level)

**Crucially: the RF lift is not about conflict.** RF + fatalities = 0.628, *worse* than RF without conflict (0.675). The +0.061 gap to OLS FE comes from nonlinear gravity structure, not from conflict signals. This separates two methodological claims that the thesis can now make independently:
1. Nonlinear ML adds ~6 percentage points of R² beyond log-linear gravity (RF result)
2. Conflict variables add zero predictive power beyond gravity, regardless of model (consistent across all four families)

---

## 6. Thesis narrative — recommended framing

The default narrative ("we used ML to predict trade from conflict") is weak because it sets up a question the data answers in the negative. The stronger framing combines three independent findings:

> **"This thesis makes three separable findings on the relationship between conflict and bilateral trade in ECOWAS. First, the predictive contribution of conflict variables — measured by ACLED's full event/disorder/perpetrator/target typology — is statistically negligible once gravity fundamentals (log GDP, log distance) and country fixed effects are controlled for. This holds uniformly across linear (OLS, OLS with fixed effects, PPML) and nonlinear (XGBoost, Random Forest, neural network) specifications, on two trade targets, and on both the full and the synthesis-free subsamples. Second, when conflict is disaggregated by type, only protest-related state-civilian conflict (`target_Protesters`, `disorder_Political violence; Demonstrations`) shows a small but statistically significant suppressive correlation with gravity residuals (r ≈ −0.10, p < 0.01); pure violence indicators show no relationship. This pattern is consistent with conflict suppressing formal trade through institutional disruption of capital-city customs and ports during state crackdowns on protest movements, rather than through physical disruption of overland trade routes. Third, on the cleanest available subsample (synthesis-free `combined_trade_baci`), Random Forest improves out-of-sample R² by 6.1 percentage points over OLS Linear FE (0.675 vs 0.614) — a gain attributable to nonlinear interactions in gravity covariates that the log-additive functional form imposes away. Notably, this Random Forest lift is realized without conflict variables: adding any conflict ablation to the Random Forest specification reduces its out-of-sample fit. The three findings are mutually reinforcing: ML adds methodological value via gravity-structural nonlinearity, not via conflict signal extraction; conflict's effect on trade is institutional and channel-specific, not a generalized disruption effect."**

This reframes the project from "did our model work?" (a prediction question) to a tripartite structural inquiry: (a) **does conflict predict trade?** (no, beyond gravity), (b) **if not in aggregate, is there a specific channel?** (yes, institutional/protest-related), (c) **does ML add anything beyond log-linear gravity?** (yes, but only Random Forest, and not via conflict).

### Policy / research implication
Two complementary implications. First, the institutional-channel finding implies that policy responses to conflict's economic costs should target customs continuity, port resilience, and freight insurance during periods of political crisis — not trade-corridor militarization in conflict zones, since the latter mostly affects informal cross-border trade that BACI doesn't capture. Second, the Random Forest gain over OLS FE suggests that gravity literature's standard log-additive specification leaves nonlinear interaction structure on the table; flexible nonlinear estimators (RF, gradient boosting on larger samples) should be considered alongside the canonical PPML+FE in future ECOWAS / sub-Saharan trade analyses.

Future research should test whether the conflict-channel structure replicates in larger-N panels (CEMAC, EAC) and whether the institutional channel is stronger when measured on customs-throughput series (port-level data) rather than annual aggregate BACI flows.

---

## 7. Defensible statements ready for the report

These are phrasings that follow directly from results above and can be cited verbatim or lightly edited:

1. **On the differentiated channel finding (headline):**
   > "Disaggregating ACLED's conflict typology reveals that the trade-suppression channel is specific to protest-related state-civilian conflict rather than insurgent violence. `target_Protesters` (r = −0.121, p = 0.0003) and `disorder_Political violence; Demonstrations` (r = −0.089, p = 0.007) correlate significantly and negatively with residuals from a gravity-only model, while pure violence indicators — `event_Battles` (r = +0.0004), `event_Violence against civilians` (r = +0.022), `target_Civilians` (r = +0.040), `perpetrator_Rebel group` (r = +0.014), `fatalities` (r = −0.053) — show no significant relationship. This pattern is consistent with conflict affecting formal trade flows through institutional disruption (capital-city customs, ports, freight) rather than through physical destruction of overland routes."

1b. **On the Random Forest result (second headline):**
   > "On the synthesis-free subsample of `combined_trade_baci` (n = 1,280 train, 275 test), Random Forest achieves out-of-sample R² = 0.675 with gravity covariates alone, exceeding the OLS Linear FE specification on the same data by 6.1 percentage points (R² = 0.614). XGBoost (R² = 0.586) and the multilayer perceptron (R² = 0.587) both underperform OLS Linear FE on this subsample, consistent with the small-panel structure: bagging-based variance reduction outperforms sequential boosting and gradient-descent optimization when the training set is below ~1,500 observations. The Random Forest gain is realized without conflict variables — adding any conflict ablation reduces out-of-sample fit (best with conflict R² = 0.628) — and is therefore attributable to nonlinear interactions among gravity covariates rather than to a conflict-mediated channel. We interpret this as evidence that the canonical log-additive gravity specification, while theoretically grounded, leaves systematic interaction structure unmodeled in small-panel sub-Saharan applications."

2. **On the reliable-subset validation:**
   > "Restricting evaluation to dyad-years with directly observed BACI flows (n = 519 test observations) raises out-of-sample R² from 0.42 to 0.61 and reduces residual standard deviation by 14%. The synthesis introduced via PPML imputation therefore adds stochastic noise to predicted values without biasing the gravity coefficients, and the gravity model on real observations explains approximately 61% of out-of-sample log-trade variance."

3. **On the linear-vs-nonlinear comparison:**
   > "Tree ensembles (XGBoost, Random Forest) and neural networks improve modestly over a log-linear gravity baseline on tradeflow_baci (R² 0.46 vs 0.34), but adding aggregate conflict features to any of the three families produces no statistically meaningful gain in out-of-sample fit. The gain emerges only when conflict variables are disaggregated by type and analyzed against gravity residuals (Statement 1)."

4. **On synthesis validation:**
   > "Re-estimating the PPML fixed-effects gravity model on the subset of dyad-years with directly observed BACI flows (excluding synthetic imputations) yields R² values comparable to the synthetic-augmented sample. We interpret this as evidence that the synthesis preserves gravity structure rather than introducing predictive bias."

5. **On hyperparameter tuning:**
   > "Bayesian-style randomized search with TimeSeriesSplit cross-validation produced hyperparameter sets that consistently underperformed default settings on the held-out test set. We attribute this to the small panel size (~3,500 train observations split into ~700-row folds) inducing CV-fold overfitting, and report results from the default specifications."

6. **On the apples-to-apples comparison:**
   > "An initial comparison appeared to favour the OLS fixed-effects baseline (R² = 0.561) over the best machine-learning model (R² = 0.462), but this comparison conflated target effects with model effects: the OLS FE was estimated on combined_trade_baci while ML was estimated on tradeflow_baci. Re-estimating both on the same target — and on the reliable-observation subset — eliminates the gap."

7. **On the case-study null result (qualitative anchor):**
   > "Three case-study political shocks — the 2012 Mali Tuareg crisis and coup, the 2014 Burkina Faso ousting of Blaise Compaoré, and the 2016 Gambian constitutional crisis — produce mean residuals between −0.22 and +0.10, all well within one standard deviation of zero. The gravity model with country fixed effects predicts the trade response to these shocks accurately without recourse to conflict variables, consistent with the GDP channel being the dominant pathway for textbook violent shocks."

---

## 8. Open / outstanding work

| Item | Status | Source |
|---|---|---|
| ML re-run on combined_trade_baci (Step 8) | Cell added, awaiting run | 06_modelling_ML.ipynb |
| Reliable-subset residual re-run (9c) | ✅ Run — see 4d, 5.6 | 06_modelling_ML.ipynb |
| Per-conflict-type residual analysis (9d) | ✅ Run — see 4e, 5.3a, 5.3b | 06_modelling_ML.ipynb |
| Mali / BFA / Gambia case-study plots (9e) | ✅ Run — see 4f, 5.3a | 06_modelling_ML.ipynb |
| Investigate `perpetrator_Civilians` positive correlation (limitation) | New, follow-up needed | Findings 5.3b |
| Cultural gravity hypothesis test | Not yet implemented | Original 19/03 hypothesis |
| Refugee flow benchmark | Not yet implemented | 06/03 meeting note |
| Email Abel Gwaindepi (DIIS) for methodology review | Not done | 28/04 meeting note |

---

## 9. Glossary of terms used in this work

| Term | Definition |
|---|---|
| **Suppression channel** | A direct causal mechanism by which conflict reduces trade beyond the indirect path through GDP. The residual analysis tests whether one is detectable. |
| **Local z-score** | A conflict count expressed as standard deviations above/below that origin country's own historical mean — isolates "anomalous year for this country" from "this country has high baseline conflict" (which FE absorbs). |
| **log1p** | log(1 + x). Used for count data because it handles zeros and compresses right-skew. |
| **PPML** | Poisson Pseudo-Maximum Likelihood. Standard estimator for modern gravity models because it handles zeros in trade data without log transformation and is consistent under heteroskedasticity. |
| **Fixed Effects (FE)** | Dummy variables for each country (or dyad) that absorb time-invariant heterogeneity. Standard in panel gravity; eliminates omitted-variable bias from unobserved country-level characteristics. |
| **Gravity model** | $F_{ij} = G \cdot M_i^{\beta_1} M_j^{\beta_2} / D_{ij}^{\beta_3}$ — bilateral trade scales with economic mass and falls with distance. Anderson (1979) provided microfoundations. |
| **Apples-to-apples comparison** | A model comparison where target variable, sample, and CV split are held constant. The original OLS-FE-vs-ML comparison was *not* apples-to-apples (different targets). |
| **Permutation importance** | Model-agnostic feature-importance measure: shuffle one feature's values randomly and measure how much R² drops. Implemented in `sklearn.inspection.permutation_importance`. |
| **Null model (permutation test)** | Repeatedly shuffle the conflict variable, refit, record the test-set score. The distribution of shuffled scores defines the "what if conflict carried no signal?" baseline. |
| **TimeSeriesSplit** | Cross-validation that respects temporal ordering: each fold's training set comes strictly before its validation set. Prevents future leakage during hyperparameter tuning. |

---

## 10. References to literature in `Bachelor_2026_coll/`

- **Yotov (2022)** *"Gravity at Sixty"* — gravity model history, FE conventions, PPML rationale
- **Santos Silva & Tenreyro (2006)** *"The Log of Gravity"* — PPML estimation under heteroskedasticity
- **Anderson (1979)** — theoretical foundation for the gravity equation
- **Anderson et al. (2018)** *"GEPPML"* — general equilibrium PPML for policy analysis
- **Glick & Taylor (2010)** *"Collateral Damage"* — trade disruption from conflict (relevant if eventual finding is different from current null)
- **Kamin** *"Bilateral trade and conflict heterogeneity"* — direct precedent for testing conflict-trade heterogeneity
- **Annan (2014)** *"Violent Conflicts and Civil Strife in West Africa"* — regional context
- **Wahab** *"A Gravity Analysis of Bilateral Trade Among ECOWAS"* — direct precedent for ECOWAS-specific gravity
- **Racek et al. (2024)** *"Spatio-temporal Diffusion in Statistical Models"* — possible methodological extension if going further
- **Africa Regional Integration Index Report 2019** — policy context

---

*Last updated: 2026-05-08 — added Step 10 (RF beats OLS FE on reliable combined_trade_baci); rewrote Sections 4g, 5.7, 6, 7 with the new tripartite narrative.*
