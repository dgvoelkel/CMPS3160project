def update_model_features(df):
    """
    Adds regression model features to a player game-log dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        The game-log dataframe.

    Returns
    -------
    pandas.DataFrame
        A copy of df with the new feature columns added.
    """

    df = df.copy()

    def get_last_6_points(df):
        df["Avg_PPG_last_6"] = (
            df
            .sort_values(["PLAYER_ID", "GAME_DATE"])
            .groupby("PLAYER_ID")["PTS"]
            .transform(lambda x: x.shift(1).rolling(6, min_periods=1).mean())
        )

        return df

    def get_matchup_average(df):
        df["Avg_PPG_Matchup"] = (
            df
            .sort_values(["PLAYER_ID", "GAME_DATE"])
            .groupby(["PLAYER_ID", "MATCHUP"])["PTS"]
            .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        )

        return df

    def get_usage_rate(df):
        team_totals = (
            df
            .groupby(["GAME_ID", "TEAM_ID"])[["FGA", "FTA", "TOV", "MIN"]]
            .sum()
            .rename(columns={
                "FGA": "TEAM_FGA",
                "FTA": "TEAM_FTA",
                "TOV": "TEAM_TOV",
                "MIN": "TEAM_MIN"
            })
            .reset_index()
        )

        sorted_df = df.sort_values(
            ["SEASON", "PLAYER_ID", "GAME_DATE"]
        ).copy()

        sorted_df["ORIGINAL_INDEX"] = sorted_df.index

        temp = sorted_df.merge(
            team_totals,
            on=["GAME_ID", "TEAM_ID"],
            how="left"
        )

        temp["GAME_USG_PCT"] = (
            100
            * (temp["FGA"] + 0.44 * temp["FTA"] + temp["TOV"])
            * (temp["TEAM_MIN"] / 5)
            / (
                temp["MIN"]
                * (temp["TEAM_FGA"] + 0.44 * temp["TEAM_FTA"] + temp["TEAM_TOV"])
            )
        )

        temp["USG_PCT"] = (
            temp
            .groupby(["SEASON", "PLAYER_ID"])["GAME_USG_PCT"]
            .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
        )

        df["USG_PCT"] = (
            temp
            .set_index("ORIGINAL_INDEX")["USG_PCT"]
            .reindex(df.index)
        )

        return df

    def get_last_6_shot_rates(df):
        sorted_df = df.sort_values(
            ["PLAYER_ID", "GAME_DATE"]
        ).copy()

        sorted_df["3PAr_last_6"] = (
            sorted_df
            .groupby("PLAYER_ID")["3PAr"]
            .transform(lambda x: x.shift(1).rolling(6, min_periods=1).mean())
        )

        sorted_df["2PAr_last_6"] = (
            sorted_df
            .groupby("PLAYER_ID")["2PAr"]
            .transform(lambda x: x.shift(1).rolling(6, min_periods=1).mean())
        )

        sorted_df["FTAr_last_6"] = (
            sorted_df
            .groupby("PLAYER_ID")["FTAr"]
            .transform(lambda x: x.shift(1).rolling(6, min_periods=1).mean())
        )

        df["3PAr_last_6"] = sorted_df["3PAr_last_6"].reindex(df.index)
        df["2PAr_last_6"] = sorted_df["2PAr_last_6"].reindex(df.index)
        df["FTAr_last_6"] = sorted_df["FTAr_last_6"].reindex(df.index)

        return df

    def get_home_away_season_avg(df, stat_col="PTS", new_col=None):
        if new_col is None:
            new_col = f"SEASON_{stat_col}_AVG_BY_HOME"

        sorted_df = df.sort_values(
            ["SEASON", "PLAYER_ID", "HOME", "GAME_DATE"]
        ).copy()

        group_cols = ["SEASON", "PLAYER_ID", "HOME"]

        prior_sum = (
            sorted_df
            .groupby(group_cols)[stat_col]
            .cumsum()
            - sorted_df[stat_col]
        )

        prior_count = (
            sorted_df
            .groupby(group_cols)
            .cumcount()
        )

        sorted_df[new_col] = prior_sum / prior_count

        df[new_col] = sorted_df[new_col].reindex(df.index)

        return df

    df = get_last_6_points(df)
    df = get_matchup_average(df)
    df = get_usage_rate(df)
    df = get_last_6_shot_rates(df)
    df = get_home_away_season_avg(df, "PTS", "HOME_AWAY_PPG")

    return df
