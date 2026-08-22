-- Run this while connected to DB_one as the Microsoft Entra administrator.
-- The name must match the Azure App Service managed identity display name.
IF NOT EXISTS (SELECT 1 FROM sys.database_principals WHERE name = N'Sam-Speed')
BEGIN
    CREATE USER [Sam-Speed] FROM EXTERNAL PROVIDER;
END;

-- Remove any legacy broad memberships before applying the narrow grants below.
IF IS_ROLEMEMBER(N'db_datareader', N'Sam-Speed') = 1
    ALTER ROLE [db_datareader] DROP MEMBER [Sam-Speed];
IF IS_ROLEMEMBER(N'db_datawriter', N'Sam-Speed') = 1
    ALTER ROLE [db_datawriter] DROP MEMBER [Sam-Speed];

GRANT SELECT, INSERT, UPDATE ON dbo.AflModelRuns TO [Sam-Speed];
GRANT SELECT, INSERT, UPDATE ON dbo.AflPredictions TO [Sam-Speed];
GRANT SELECT, INSERT, UPDATE ON dbo.AflPredictionSnapshots TO [Sam-Speed];
