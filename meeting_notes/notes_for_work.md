### 04/03/2026
Based on initial EDA, we found multiple columns which had a large amount of NaN rows. These would cause issues when working with continuous values.
The main culprits came from a few, very well-represented rows:
col_dep_end_conflict - 282284 NaNs
rta_type - 267688 NaNs
sever_year - 234470 NaNs
tradeflow_comtrade_o - 195722
tradeflow_imf_o - 178318
tradeflow_comtrade_d - 160611
tradeflow_imf_d - 157168
tradeflow_baci - 147439   (KEPT)
manuf_tradeflow_baci - 147439   (KEPT)
entry_tp_d -  136903
entry_tp_o - 101167
scaled_sci_2021 - 81335


These features were dropped based on the amount of missing rows in the overall dataset. 


Problem: The Gravity dataset is so broad and general, that the conclusions are almost self-explanatory (lower gdp, more conflict)
In addition, we have few hypotheses that can be made specific from it. (How does the Gambian constitutional crisis of 2016 affect their relations with neighboring ECOWAS countries?) leads to a very general uninteresting point. The riots are significant, but can such a ML model be expanded to more?

Instead, perhaps we need to narrow down our focus on products in the BACI dataset. Here we can find a more tangible effect on specifics. For example, what amount of conflict in the highest cocoa producing countries could collapse the trade? 
We should ask "what" questions rather than "why" questions. What happens when conflict erupts in a specific country to trade in very specific wares? Can we create a hypothetical scenario, where a trained model can find the impact on such production in the future? 






### --- 09/03/2026 ---

There are certain interesting ideas for international trade we can take into account.
https://www.sciencedirect.com/science/article/pii/S0022199610000036  - The erosion of colonial trade linkages after independence (2010)

https://pure.au.dk/ws/files/55766245/wp13_13.pdf - Revisiting the Effectiveness of African Economic Integration. A Meta-Analytic Review and Comparative Estimation Methods (2013)

Maybe it is relevant to look at the Heckscher–Ohlin model for bilateral trade. (https://en.wikipedia.org/wiki/Heckscher%E2%80%93Ohlin_model)


It is generally held that the Gravity Model of Trade is true. 
$F_{ij} = G \frac{M_i^{\beta_1} M_j^{\beta_2}}{D_{ij}^{\beta_3}} \eta_{ij}$


where $F_{ij}$ represents the volume of trade from country $i$ to country $j$,  
$M_i$ and $M_j$ represent the GDPs of countries $i$ and $j$,  
$D_{ij}$ denotes the distance between them,  
and $\eta_{ij}$ is an error term with expectation equal to 1.

So perhaps we can compare this model to our own findings, to more adequately prove that the conflict in a given country is affecting trade. 


Since we are dealing with a combined panel structure (longitudinal data) in our data, we can use PanelOLS in our initial data, then later adding scikit-learn, once we want to add regression models to the data, to train a ML-model. 


Hypothesis could be updated to something akin to:
How do changes in violence within a country over time affect changes in trade within the same country?

We are looking at changes!




### --- 12/03/2026 ---
_Theoretical breakthroughs, or how I learned to start worrying about dyadic heterogeneity and heteroscedastic characteristics_

We can consider the model we are trying to work on as a linear relationship between the dependant variable Trade, and the many independent variables in conflicts, etc.

One issue with our previous EDA, was the Pearson scores we were getting implied a homoscedastic relationship in the overall error terms. However, our variables are almost certainly heteroscedastic (can be tested, see below), this was a faulty premise.

We should be able to find a solution in using Ordinary Least Squares (OLS), but we should consider things that are specific for use with our case: Panel data (multiple entities over time) that account for certain fixed effects (that would be the distance between countries, historical ties, economic unions etc.) -> This leads us to Panel OLS with Fixed Effects.
We are most likely dealing with data that is not homoskedastic (there are simply too many variables involved), and we can prove this using the Breusch–Pagan test, which is a chi-squared test for heteroskedacity. (_Should we?_)

Secondly, this leads to something called dyad fixed effects, which are common in gravity models of trade. 

We are trying to implement LabelOLS to find the F-values of the variables in the ACLED dataset. However, we have issues when dealing with too many variables in the same model at once, since "_When your regressors are highly collinear and vary only over time (as in your setup where ACLED is origin-only), you can’t throw them all into one FE model without running into rank/absorption problems._"

Therefore, we have to find a way to put the variables up against each other. We can:
    "_Use the Frisch–Waugh–Lovell (FWL) trick to get each variable’s within effect with entity FE_:

    - Residualize the dependent and each candidate variable against entity dummies.
    - Run a simple OLS of residualized y on residualized x (one variable at a time)."
Secondly, we must find a way to deal with heteroscedastic data. Is it possible to solve this using Weighted least squares? (https://en.wikipedia.org/wiki/Weighted_least_squares) (also see https://en.wikipedia.org/wiki/Heteroskedasticity-consistent_standard_errors for more information)
Otherwise, Fixed Effects seem to be the overall best solution to the problem.


The reason we can not use a simple pooled OLS is that the full mathematical formula would be $Trade_{ijt} = β*Conflict_{it} + α_{ij} + γ_{t} + u_{ijt}$ where $α_{ij}$ corresponds to all stable dyad pair characteristics (this is  time-invariant heterogeneity. Things that do not change between countries, but affect trade links) and $γ_{t}$ are the yearly effects, such as global price shocks.

The pooled OLS would ignore the input of the heterogeneous characteristics, and would thus lead to a very biased model. Therefore our previous choice of simply applying a Pearson score does not adequately account for all the variables we are dealing with. 

_Dyad fixed effects include a separate intercept for each dyad (or equivalently, “within transform” / demeaning)_. That is to say, a Dyad FE is in place to remove $α_{ij}$ entirely from the equation. Specifically the things that do not change over time. 

A third thought we have included in the handling of trade, is whether to log trade in order to handle any extreme outliers. OLS also assumes normally distributed residuals, but as trade can fluctuate wildly between countries, we can make things more symmetric by logging it. 
The logarithmic approach should also ideally reduce the heteroskedasticity, as the larger variance of larger economies gets squeezed together, and the smaller differences in smaller economies get expanded. 
The gravity equation, as detailed above, is also typically used with the log scale as a standard, as we are usually dealing with percentages that are easily captured in the log-scale models. So for a small economy, going from 10M to 20M leads to a 100% increase, akin to going from 10B to 20B, even though the numbers are much smaller. 

Issues with logging are how to deal with zero values (which we have many of in the dataset. How do we deal with NaNs in the best way?) One solution is to use the Poisson Pseudo-Maximum Likelihood (look it up! Standard use for modern Gravity Theory uses)



### --- 19/03/2026 ---
We have an aggregate system of dataframes for all members of ECOWAS that can be worked on interchangably. We can access these through a dictionary.
In addition, we now have a new, improved hypothesis that can synthesize the economic and cultural elements of the project:

As a result of conflicts, do trade connections between countries sharing historical and linguistic backgrounds conserve a stronger tradeflow than countries where these differ, and (as an appendum), are refugees from a conflict country more likely to seek out countries with higher cultural gravity?

In ECOWAS, we have three languages represented and four previous cultural hegemons: 
English (Nigeria, Ghana, Sierra Leone, The Gambia)
French (Benin, Guinea, Ivory Coast, Senegal, Togo, Mali, Niger, Burkina Faso)
Portuguese (Cape Verde, Guinea-Bissau)
US (Liberia)

As we have a high number of NaN values for Liberia and the Portuguese-speaking nations, we are opting to exclude these in the initial stages, to get useful results with the remaining nations.


Discovery of Gravity:
    Certain missing values NaN in tradeflow can be found in the BACI tradesets. (for example Gambia-Guinea 2021)
    Tradeflow in Gravity is specifically the export from origin country to destination in the dyad. Therefore all values in our Gambia_df is strictly exports
    Gambia's exports to other ECOWAS countries grow massively around 2010, before falling back down. We can look into the BACI-set to find out what constitutes this growth.

    Time to create a new column that makes a combined trade dataset.



### --- 03/04/2026 ---
For the future, think of graph neural networks. Might be very applicable to our work

Major question:
HOW are we going to synthesise data? First idea is to simply use the Gravity Model: $F_{ij} = G \frac{M_i^{\beta_1} M_j^{\beta_2}}{D_{ij}^{\beta_3}} \eta_{ij}$

where $F_{ij}$ represents the volume of trade from country $i$ to country $j$,  
$M_i$ and $M_j$ represent the GDPs of countries $i$ and $j$,  
$D_{ij}$ denotes the distance between them,  
and $\eta_{ij}$ is an error term with expectation equal to 1.


In our case, it would be Gravity(year/Country)->Gravity(GDP_o)*Gravity(GDP_d) / Gravity(distance)
However, we also need to find the elasticities of exports and imports (these are the beta values we are raising the values to), and a constant G which will work as our overall scaling constant 

In other words, we estimate elasticities of exporter size, importer size, and distance, and the constant term G from a PPML gravity regression. Then we plug these into the multiplicative gravity equation to generate synthetic trade flows.


Since Poisson regression and distributions are good for count data, it is a generally good approach for dealing with the kind of data we have.
However, after implementing an initial poisson with the fixed effects (exporter, importer, year) and working on pure raw GDP, aiming to synthesize tradeflow_baci for every dyad, we found that the synthetic data was wildly inaccurate. 
Once we removed the fixed effects and ran it on GDP Per Capita, it was however wildly inaccurate for the larger economies, as it did not adequately account for changes in absolute size of the economies.


Another note to make regarding our countries, is that tradeflow_baci is specifically only tradeflows in terms of goods. We also have trade in services and other economic movements that are defined in other columns like (tradeflow_comtrade_o, tradeflow_comtrade_d, tradeflow_imf_o, tradeflow_imf_d). And tradeflow_baci is specifically only the exports in each country. Should we be combining economic data from all of these when looking at the developments? And should we add synthetic data to all of these? 


--- 06/04/2026 --- 
A growing issue is the bad documentation of trade for the African countries. Trade might be mislabelled, if not be outright missing. The problem isn't necessarily major in our case, as we are only interested in the effect conflict has on trade (thus we can extrapolate the differences in reported trade to overall trade (though taking in mind, changes in bureaucratic practice in countries could more heavily affect trade than anything else)). However, we propose a combined trade activity column, that for each dyad take the combined BACI, IMF and Comtrade columns. However, even with these additions, we still have some dyads that are missing trade information.





