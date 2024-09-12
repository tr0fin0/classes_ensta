
Drop TABLE DET;
Drop TABLE PRO;
Drop TABLE COM;
Drop TABLE FOU;
Drop TABLE CLI;

CREATE TABLE CLI(
  NumCli number(5) Constraint PK_Cli PRIMARY KEY,
  NomCli char(20),
  Pays char(30),
  Tel char(15));

CREATE TABLE FOU(
  NumFou number(2) Constraint PK_FOU PRIMARY KEY, 
  NomFou char(20),
  Pays char(30), 
  Tel char(15));

CREATE TABLE COM(
  NumCom number(5) Constraint PK_COM PRIMARY KEY,
  NumCli Number(5) Constraint COM_REF_CLI REFERENCES Cli ,
  FraisPort number(4),
  AnCom number(4));

CREATE TABLE PRO(
  NumPro number(5) Constraint PK_PRO PRIMARY KEY,
  NumFou number(2) Constraint PRO_REF_FOU REFERENCES Fou,
  NomPro char(20),
  TypePro char(10),
  PrixUnit number(3));

CREATE TABLE DET(
  NumCom number(5) Constraint DET_REF_COM REFERENCES Com ,
  NumPro number(5) Constraint DET_REF_PRO REFERENCES Pro,
  Qte number(5),
  Remise number(5),
  Constraint PK_DET PRIMARY KEY (NumCom, NumPro));

