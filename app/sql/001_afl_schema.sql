IF OBJECT_ID(N'dbo.AflModelRuns', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.AflModelRuns (
        model_version NVARCHAR(64) NOT NULL PRIMARY KEY,
        trained_at DATETIMEOFFSET NOT NULL,
        holdout_season SMALLINT NOT NULL,
        training_matches INT NOT NULL,
        holdout_tip_accuracy DECIMAL(7,6) NULL,
        holdout_margin_mae DECIMAL(8,3) NULL,
        report_json NVARCHAR(MAX) NOT NULL,
        created_at DATETIMEOFFSET NOT NULL DEFAULT SYSUTCDATETIME(),
        CONSTRAINT CK_AflModelRuns_ReportJson CHECK (ISJSON(report_json) = 1)
    );
END;

IF OBJECT_ID(N'dbo.AflPredictionSnapshots', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.AflPredictionSnapshots (
        snapshot_id NVARCHAR(128) NOT NULL PRIMARY KEY,
        model_version NVARCHAR(64) NOT NULL,
        season SMALLINT NULL,
        round_name NVARCHAR(80) NOT NULL,
        prediction_count SMALLINT NOT NULL,
        generated_at DATETIMEOFFSET NOT NULL,
        payload_json NVARCHAR(MAX) NOT NULL,
        created_at DATETIMEOFFSET NOT NULL DEFAULT SYSUTCDATETIME(),
        CONSTRAINT FK_AflPredictionSnapshots_ModelRun FOREIGN KEY (model_version)
            REFERENCES dbo.AflModelRuns(model_version),
        CONSTRAINT CK_AflPredictionSnapshots_PayloadJson CHECK (ISJSON(payload_json) = 1)
    );
    CREATE INDEX IX_AflPredictionSnapshots_GeneratedAt
        ON dbo.AflPredictionSnapshots(generated_at DESC);
END;

IF OBJECT_ID(N'dbo.AflPredictions', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.AflPredictions (
        game_id NVARCHAR(40) NOT NULL PRIMARY KEY,
        model_version NVARCHAR(64) NOT NULL,
        season SMALLINT NOT NULL,
        round_number SMALLINT NOT NULL,
        round_name NVARCHAR(80) NOT NULL,
        start_time DATETIMEOFFSET NOT NULL,
        venue NVARCHAR(120) NOT NULL,
        home_team NVARCHAR(80) NOT NULL,
        away_team NVARCHAR(80) NOT NULL,
        home_win_probability DECIMAL(6,3) NOT NULL,
        away_win_probability DECIMAL(6,3) NOT NULL,
        draw_probability DECIMAL(6,3) NOT NULL,
        expected_home_margin DECIMAL(8,3) NOT NULL,
        interval_80_low DECIMAL(8,3) NULL,
        interval_80_high DECIMAL(8,3) NULL,
        factors_json NVARCHAR(MAX) NOT NULL,
        generated_at DATETIMEOFFSET NOT NULL,
        updated_at DATETIMEOFFSET NOT NULL DEFAULT SYSUTCDATETIME(),
        CONSTRAINT FK_AflPredictions_ModelRun FOREIGN KEY (model_version)
            REFERENCES dbo.AflModelRuns(model_version),
        CONSTRAINT CK_AflPredictions_FactorsJson CHECK (ISJSON(factors_json) = 1),
        CONSTRAINT CK_AflPredictions_HomeProbability CHECK
            (home_win_probability >= 0 AND home_win_probability <= 100),
        CONSTRAINT CK_AflPredictions_AwayProbability CHECK
            (away_win_probability >= 0 AND away_win_probability <= 100),
        CONSTRAINT CK_AflPredictions_DrawProbability CHECK
            (draw_probability >= 0 AND draw_probability <= 100),
        CONSTRAINT CK_AflPredictions_ProbabilityTotal CHECK
            (ABS(home_win_probability + away_win_probability + draw_probability - 100) <= 0.101)
    );
    CREATE INDEX IX_AflPredictions_StartTime ON dbo.AflPredictions(start_time);
END;
