within Credibility.Types;
record IntervalTolerance1D "Interval uncertainty of lambda for 1D table with tolerance (epistemic uncertainty)"
  extends BaseUncertainty1D(
    kind = UncertaintyKind1D.IntervalTolerance1D);

  constant Boolean withTolerance=true "= true, if uncertainty defined by tolerance";
  parameter Real table[:,2] = zeros(1,2)
    "Table matrix (columns: 1 = input/grid; 2 = nominal values)";
  parameter Real relTol=0.0
    "Limits: |value[i] - table[i,2]| <= max(absTol,relTol*|table[i,2]|)";
  parameter Real absTol=0.0
    "Limits: |value[i] - table[i,2]| <= max(absTol,relTol*|table[i,2]|)";

  annotation (Documentation(info="<html>
<p>
An <em>epistemic</em> uncertainty of 1d table, i.e. the uncertainty <em>due to lack
of knowledge</em> &ndash; e.g., on the side of model, analysis or experiment.
The uncertainty is described by a&nbsp;<em>nominal value</em> and an
<em>interval</em>.
See also <a href=\"modelica://Credibility.Types.BaseUncertainty1D\">BaseUncertainty1D</a>
uncertainty for general information.
</p>


<h4>See also</h4>
<ul>
  <li>
    <a href=\"modelica://Credibility.Types.IntervalTolerance\">IntervalTolerance</a>
    for additional information.
  </li>
  <li>
    <a href=\"modelica://Credibility.Utilities.Table1DScalings.getTableLambdaByTolerance\">getTableLambdaByTolerance</a>
    for usability.
  </li>
</ul>
</html>"));
end IntervalTolerance1D;
