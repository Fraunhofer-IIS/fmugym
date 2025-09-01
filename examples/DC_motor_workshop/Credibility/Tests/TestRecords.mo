within Credibility.Tests;
model TestRecords "Test of all records defined in the library"
    extends Modelica.Icons.Example;

    Credibility.Types.Interval interval(
      source=Credibility.Types.SourceType.Unknown,
      unitValue="kg",
      lower=8.0,
      upper=12.0) annotation (Placement(transformation(extent={{-90,60},{-70,80}})));
    Credibility.Types.Uniform uniform(
      source=Credibility.Types.SourceType.Estimated,
      unitValue="kg",
      lower=8.0,
      upper=12.0) annotation (Placement(transformation(extent={{-60,60},{-40,80}})));
    Credibility.Types.TruncatedNormal truncatedNormal(
      source=Credibility.Types.SourceType.Provided,
      unitValue="kg",
      lower=7.0,
      upper=13.0,
      mean=10.0,
      stdDev=1.5) annotation (Placement(transformation(extent={{-30,60},{-10,80}})));
    Credibility.Types.IntervalTolerance intervalTolerance(
      source=Credibility.Types.SourceType.Computed,
      unitValue="kg",
      nominal=10.0,
      relTol=0.2) annotation (Placement(transformation(extent={{0,60},{20,80}})));
    Credibility.Types.UniformTolerance uniformTolerance(
      source=Credibility.Types.SourceType.Measured,
      unitValue="kg",
      nominal=10.0,
      absTol=2.0) annotation (Placement(transformation(extent={{30,60},{50,80}})));
    Credibility.Types.TruncatedNormalTolerance truncatedNormalTolerance(
      source=Credibility.Types.SourceType.Calibrated,
      unitValue="kg",
      nominal=15.0,
      relTol=0.2,
      stdDevFactor=2) annotation (Placement(transformation(extent={{60,60},{80,80}})));
    Credibility.Scalar scalar1(value(unit="kg") = 10.0)
      annotation (Placement(transformation(extent={{-60,20},{-40,40}})));
    Credibility.Scalar scalar2(
      value(unit="kg") = 10.0,
      traceability(reference=""),
      redeclare Credibility.Types.TruncatedNormalTolerance
                                                     uncertainty(
        source=Credibility.Types.SourceType.Estimated,
        unitValue="kg",
        nominal=10.0,
        relTol=0.2),
      redeclare Credibility.Types.Calibration
                                  calibration(
        unitValue="kg",
        start=9,
        lower=5,
        upper=15))
      annotation (Placement(transformation(extent={{-20,20},{0,40}})));
    Credibility.Table1D table1D(redeclare Credibility.Types.Interval1D uncertainty(lambda=0.1, table=[0,100,90,110; 1,120,115,125; 2,115,110,123; 3,100,90,110; 4,70,50,90; 5,-10,-38,18])) annotation (Placement(transformation(extent={{-60,-60},{-40,-40}})));

    Credibility.Types.Interval1D interval1D(lambda=0.8, table=[0,2,1,3; 1,3,2,4; 4,4,3,5]) annotation (Placement(transformation(extent={{-90,-20},{-70,0}})));
    Credibility.Types.Uniform1D uniform1D(
      source=Credibility.Types.SourceType.Provided,
      lambda=-0.1,
      table=[0,2,1,3; 1,3,2,4; 4,4,3,5]) annotation (Placement(transformation(extent={{-60,-20},{-40,0}})));
    Credibility.Types.TruncatedNormal1D truncatedNormal1D(
      source=Credibility.Types.SourceType.Computed,
      lambda=0.2,
      table=[0,2,1,3; 1,3,2,4; 4,4,3,5],
      stdDev=3) annotation (Placement(transformation(extent={{-30,-20},{-10,0}})));
    Credibility.Types.IntervalTolerance1D intervalTolerance1D(table=[0,1; 1,2; 2,4; 3,9], relTol=0.05) annotation (Placement(transformation(extent={{0,-20},{20,0}})));
    Credibility.Types.UniformTolerance1D uniformTolerance1D(table=[0,1; 1,2; 2,4; 3,9], absTol=0.1) annotation (Placement(transformation(extent={{30,-20},{50,0}})));
    Credibility.Types.TruncatedNormalTolerance1D truncatedNormalTolerance1D(
      source=Credibility.Types.SourceType.Calibrated,
      info="",
      lambda=-0.2,
      table=[0,1; 1,2; 2,4; 3,9],
      relTol=0.05,
      absTol=0.1,
      stdDevFactor=2) annotation (Placement(transformation(extent={{60,-20},{80,0}})));
    Credibility.Table1D table1D1(traceability(source=Credibility.Types.SourceType.Computed), redeclare Credibility.Types.TruncatedNormalTolerance1D uncertainty(relTol=0.5, absTol=0.01)) annotation (Placement(transformation(extent={{-20,-60},{0,-40}})));
    Credibility.Table1D table1D2 annotation (Placement(transformation(extent={{20,-60},{40,-40}})));
    annotation (
      experiment(StopTime=0.1),
      Icon(
        coordinateSystem(preserveAspectRatio=false)),
      Diagram(
        coordinateSystem(preserveAspectRatio=false)),
      Documentation(info="<html>
<p>
Test different variants of credibility records.
</p>
</html>"));
end TestRecords;
