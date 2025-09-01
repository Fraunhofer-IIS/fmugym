within Credibility.Examples.SimpleControlledDriveNonlinear;
record DataVariant001 "Data set of model variant 001"
  extends PartialData(
    variantName="DataVariant001",
    Ja(
      value = 2,
      traceability(
        source = Credibility.Types.SourceType.Provided,
        info = "Data taken from data sheet"),
      redeclare Credibility.Types.IntervalTolerance uncertainty(
        unitValue="kg.m2",
        nominal=2,
        relTol=0.05,
        source=Credibility.Types.SourceType.Provided)),
    Jm(
      value = 1,
      traceability(
        source = Credibility.Types.SourceType.Computed,
        info = "Data computed form CAD data"),
      redeclare Credibility.Types.UniformTolerance uncertainty(
        source=Credibility.Types.SourceType.Estimated,
        unitValue="kg.m2",
        nominal=1,
        relTol=0.03)),
    ratio(
      value = 2,
      traceability(
        source = Credibility.Types.SourceType.Provided,
        info = "Data taken from data sheet"),
      redeclare Credibility.Types.IntervalTolerance uncertainty(
        unitValue="1",
        nominal=2,
        relTol=0,
        source=Credibility.Types.SourceType.Provided)),
    damping(
      value = 99.78375,
      traceability(
        source = Credibility.Types.SourceType.Calibrated,
        info = "Calibrated from virtual measurements with noise",
        reference = "Credibility.Examples.SimpleControlledDriveNonlinear.SimpleControlledDrive_virtualMeasurement"),
      redeclare Credibility.Types.TruncatedNormalTolerance uncertainty(
        unitValue="N.m.s/rad",
        nominal=99.78375,
        relTol=0.2),
      redeclare Credibility.Types.Calibration calibration(
        unitValue="N.m.s/rad",
        start = 300,
        lower = 1,
        upper = 1000,
        setup = "CredibilityCalibration.Examples.SimpleControlledDriveNonlinear.runSimpleControlledDriveOptNLSetup")),
    stiffness(
      traceability(
        source = Credibility.Types.SourceType.Measured,
        info = "Based on measurements on test bed B in project XYZ."),
      uncertainty(
        source=Credibility.Types.SourceType.Estimated,
        table=[
          0.0000, 1449.6, 1232.1, 1667.1;
          0.0005, 1451.0, 1232.9, 1669.0;
          0.0010, 1453.2, 1235.4, 1671.0;
          0.0015, 1455.4, 1237.3, 1673.6;
          0.0020, 1457.8, 1237.9, 1677.7;
          0.0025, 1460.8, 1242.1, 1679.5;
          0.0030, 1465.8, 1248.9, 1682.6;
          0.0035, 1468.4, 1247.0, 1689.7;
          0.0040, 1472.2, 1256.2, 1688.3;
          0.0045, 1476.8, 1260.5, 1693.0;
          0.0050, 1482.0, 1264.1, 1700.0;
          0.0055, 1486.1, 1267.4, 1704.8;
          0.0060, 1492.5, 1273.2, 1711.7;
          0.0065, 1496.7, 1278.1, 1715.3;
          0.0070, 1501.5, 1283.6, 1719.5;
          0.0075, 1509.9, 1291.1, 1728.6;
          0.0080, 1514.9, 1297.6, 1732.2;
          0.0085, 1521.9, 1304.8, 1739.0;
          0.0090, 1528.6, 1309.0, 1748.2;
          0.0095, 1536.4, 1319.0, 1753.8;
          0.0100, 1543.6, 1324.1, 1763.1])));

  annotation (
    Documentation(info="<html>
<p>
A&nbsp;record containing particular values and information of all credible
parameters of the
<a href=\"modelica://Credibility.Examples.SimpleControlledDriveNonlinear.SimpleControlledDrive_original\">simple controlled drive example</a>.
</p>
</html>"));
end DataVariant001;
