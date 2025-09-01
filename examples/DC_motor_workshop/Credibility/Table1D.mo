within Credibility;
record Table1D "Record collecting credibility information for a Real 1D table"
  extends Credibility.Icons.CredibleBase;

  final parameter Real table[size(uncertainty.table,1),2]=
    if uncertainty.withTolerance then
      Utilities.Table1DScalings.getTableLambdaByTolerance(
        uncertainty.lambda,
        uncertainty.table[:,1:2],
        uncertainty.relTol,
        uncertainty.absTol)
    else
      Utilities.Table1DScalings.getTableLambdaByInterval(
        uncertainty.lambda, uncertainty.table)
    "Scaled table";
  parameter Types.Traceability traceability "Traceability";
  replaceable parameter Types.Interval1D uncertainty
    constrainedby Types.BaseUncertainty1D "Uncertainty"
    annotation (choicesAllMatching=true);
  parameter String calibration = "" "URI of calibration setup script";

  annotation (
    Icon(graphics={
        Text(
          extent={{-80,-10},{100,-60}},
          textColor={0,140,72},
          textString="table 1D")}),
    Documentation(info="<html>
<p>
This record collects credibility information for a&nbsp;<em>1d table</em>
parameter. In particular, it contains information regarding
</p>
<ul>
  <li>
    the nominal values as [:,2] array (first column: abscissa, second column:
    ordinate values), see also
    <a href=\"modelica://Credibility.UsersGuide.ParameterCredibility.UncertaintyInfo\">User&apos;s Guide</a>
    &ndash; &quot;Array uncertainty&quot;.
  </li>
  <li>
    traceability,
  </li>
  <li>
    uncertainty and
  </li>
  <li>
    calibration.
  </li>
</ul>
</html>"));
end Table1D;
