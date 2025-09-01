within Credibility.Types;
record TruncatedNormalTolerance1D "Truncated normal distribution of lambda for 1D table with tolerance"
  extends IntervalTolerance1D(
    final kind = UncertaintyKind1D.TruncatedNormalTolerance1D);

  parameter Real stdDevFactor=3
    "Standard deviation = max(absTol,relTol*|nominal|)/stdDevFactor"
    annotation(choices(
        choice=1 "1 (68.2%)",
        choice=2 "2 (95.4%)",
        choice=3 "3 (99.7%)",
        choice=4 "4 (99.99%)",
        choice=5 "5 (99.9999%)"));

  annotation (Documentation(info="<html>
<p>
A&nbsp;<em>truncated normally distributed</em> uncertainty of 1d table. The minimum
and the maximum possible values of the table are described by
a&nbsp;<em>nominal value</em> and an <em>interval</em>.
See also <a href=\"modelica://Credibility.Types.BaseUncertainty1D\">BaseUncertainty1D</a>
uncertainty for general information.
</p>


<h4>See also</h4>
<ul>
  <li>
    <a href=\"modelica://Credibility.Types.TruncatedNormalTolerance\">TruncatedNormalTolerance</a>
    for additional information.
  </li>
  <li>
    <a href=\"modelica://Credibility.Utilities.Table1DScalings.getTableLambdaByTolerance\">getTableLambdaByTolerance</a>
    for usability.
  </li>
</ul>
</html>"));
end TruncatedNormalTolerance1D;
