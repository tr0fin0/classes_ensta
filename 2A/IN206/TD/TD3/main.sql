-- ======================================
-- seance 20/10/2022
-- ======================================



-- question 1
-- ======================================
-- declare RESULTAT table
CREATE TABLE RESULTAT(
    code NUMBER,
    message VARCHAR(50)
);

-- clean RESULTAT table
DELETE * FROM RESULTAT;

-- input of n
PROMT Nombre de lignes a produire
    ACCEPT n

BEGIN
    FOR i IN 1..&n LOOP
        IF MOD(x, 2) = 0 THEN
            INSERT INTO RESULTAT VALUES (i, x || ' is even');
        ELSE
            INSERT INTO RESULTAT VALUES (i, x || ' is odd');
        END IF;
        x := x + 1;
    END LOOP;
COMMIT;

/

SELECT * FROM RESULTAT;

/

-- question 2
-- ======================================
-- TODO check questions

-- question 3
-- ======================================
-- TODO check questions

-- question 4
-- ======================================
-- TODO fix
SELECT numcle, nomcli, SUM(DET.qte)
FROM CLI, COM, DET
WHERE
    CLI.numcli = COM.numcli AND
    COM.numcli = DET.numcan
GROUP BY CLI.numcli;

-- question 4
-- ======================================
-- TODO check questions

-- question 5
-- ======================================
-- TODO check questions

-- question 6
-- ======================================
-- TODO check questions

-- question 7
-- ======================================
-- TODO check questions

-- question 8
-- ======================================
-- TODO check questions
