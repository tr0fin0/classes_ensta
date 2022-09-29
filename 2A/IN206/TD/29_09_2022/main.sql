-- =================
-- inicial setup
-- =================

sqlplus sys/oracle as syoba;
ALTER SESSION SET CONTAINER = XEPOB1;
CREATE USER TP IDINTIFIED BY oracle;
GRANT ALL PRIVILLEGES TO TP;
escit;


-- =====================
-- questions solutions
-- =====================

-- question 2
CREATE TABLE CLI (
    CODE_CLI NUMBER(5)
    CONSTRAINT KEY_CLI PRIMARY KEY,
    NOM_CLI CHAR(20),
    PAYS    CHAR(20)
);

-- question 3
-- yes, search why

-- question 4
-- 

-- question 5
ALTER TABLE CLI MODIFY (
    NOM_CP CHAR(30),
);

-- question 6
ALTER TABLE CLI ADD (
    TEF NUMBER(10),
);

-- question 7
ALTER TABLE PROD ADD (
    PRIX_UNIT NUMBER(10),
);

-- question 8
ALTER TABLE PROD MODIFY (
    NOM_PROD CHAR(50) CONSTRAINT
    NOM_PROD NOT NULL 
);

-- question 9
INSERT INTO CLI VALUES (
    12345, 'Toto', 'France', 0000000000
);

-- question 10
UPDATE CLI SET NOM_CLI = UPPER(NOM_CLI);

-- question 11
DROP TABLE DET PROD COM CLI

-- question 12
-- terminal
-- cd Téléchargements
-- sqlplus TP/oracle@localhost:1521/XEPOB1
-- @CREATE.sql
-- commit;
-- escit;
-- sqldr TP/oracle@localhost:1521/XEPOB1
-- <file>.ctp
-- <file>.log
-- code CLI.ctl COM.ctl COM.log