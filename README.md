NHL Predictions

Despite the wide availability of team and player-level statistics, NHL games remain notoriously difficult to predict with confidence. The typical win rate for underdog teams sits just below 40%. Compared to other mainstream sports, the point spread in hockey is narrow, leading to a higher frequency of close games. Additionally, penalties and injuries often disrupt lineup predictability. In my preliminary analysis, one of the most significant influences on pre-closing line betting odds was the announcement of the starting goalie, which could shift the odds by 5% or more.

Sportsbooks have a considerable advantage when it comes to accounting for last-minute lineup changes. As a result, their predictions are typically sharpest at the closing line and weakest at the opening line. I set out to build a model that is agnostic to these last-minute variables and instead relies solely on data available at the conclusion of each team’s previous game.

MoneyPuck, a well-known hockey analytics website, publishes a predictive model that achieves 60.4% accuracy with a log loss of 0.658. I used these metrics as a benchmark for success, as their model performs similarly to sharp sportsbook lines. In addition to their forecasts, MoneyPuck provides team-level data dating back to 2007, which is updated nightly during the regular season. While they also host downloadable CSVs for goalie and skater statistics, those datasets are incomplete. For goalie metrics, I supplemented data using the official NHL API.

To prepare the training data, I excluded a significant portion of historical games. Models trained on data from earlier seasons performed poorly when applied to current games. Ultimately, I limited training to just two recent seasons (2022–2025), as including older data resulted in a rapid decline in accuracy. I attribute this to changes in league rules over time and the anomalous nature of the 2020–2021 season due to COVID-related disruptions.

Both MoneyPuck and NHL API data are recorded on a per-game basis, which offers limited predictive power on its own. A major component of this project involved restructuring the data by:

    Separating home and away teams
    Converting game stats into rolling averages
    Recombining teams by stat disparity to simulate the upcoming matchup

This approach initially yielded about 58% accuracy before model tuning and ensembling. The final model architecture uses a logistic regression stacker on top of XGBoost and a Random Forest Regressor, with each base model tuned via grid search (which is preserved in the final code). The stacked model is then calibrated using isotonic regression to improve probability estimates.
Final Model Metrics

    Brier Score: 0.2205
    Accuracy: 65.13%
    AUC: 0.699
    Log Loss: 0.645

In tuning and calibrating the model, I prioritized log loss and AUC over raw accuracy, since the final outputs are used to calculate expected edge and betting equity. I'm satisfied with these results, though there is still room for improvement—particularly through the incorporation of schedule-based features such as rest days, winning streaks, and time away from home.



