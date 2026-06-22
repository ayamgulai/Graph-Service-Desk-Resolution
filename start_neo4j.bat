@echo off
REM ============================================================
REM  Neo4j + APOC Setup via Docker (Windows)
REM  Run AFTER Docker Desktop is fully started
REM ============================================================

echo [1/4] Stopping and removing old container (if any)...
docker stop neo4j-servicedesk 2>nul
docker rm   neo4j-servicedesk 2>nul

echo.
echo [2/4] Pulling Neo4j 5 Community Edition image...
docker pull neo4j:5-community

echo.
echo [3/4] Starting Neo4j container with APOC enabled...
docker run --detach ^
  --name neo4j-servicedesk ^
  --publish 7474:7474 ^
  --publish 7687:7687 ^
  --env NEO4J_AUTH=neo4j/ServiceDesk2024! ^
  --env NEO4J_PLUGINS=[\"apoc\"] ^
  --env NEO4J_apoc_export_file_enabled=true ^
  --env NEO4J_apoc_import_file_enabled=true ^
  --env NEO4J_apoc_import_file_use__neo4j__config=true ^
  --env NEO4J_dbms_security_procedures_unrestricted=apoc.* ^
  --env NEO4J_dbms_security_procedures_allowlist=apoc.* ^
  --volume neo4j-servicedesk-data:/data ^
  neo4j:5-community

echo.
echo [4/4] Waiting 30 seconds for Neo4j + APOC to initialize...
timeout /t 30 /nobreak

echo.
echo ============================================================
echo  Neo4j is ready!
echo  Browser UI  : http://localhost:7474
echo  Bolt URI    : bolt://localhost:7687
echo  Username    : neo4j
echo  Password    : ServiceDesk2024!
echo ============================================================
echo  To verify APOC loaded, open the browser UI and run:
echo    RETURN apoc.version()
echo ============================================================
pause
