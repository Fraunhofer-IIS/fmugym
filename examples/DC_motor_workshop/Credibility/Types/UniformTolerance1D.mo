within Credibility.Types;
record UniformTolerance1D "Uniform distribution of lambda for 1D table with tolerance"
  extends IntervalTolerance1D(
    final kind = UncertaintyKind1D.UniformTolerance1D);

  annotation (Documentation(info="<html>
<p>
An uncertainty describing possible values of the uncertain 1d table by a&nbsp;<em>uniform
distribution</em>, i.e. the probability of values&apos; outcome is equal between the
limits which are given by a&nbsp;<em>nominal values</em> and an
<em>interval</em>.
See also <a href=\"modelica://Credibility.Types.BaseUncertainty1D\">BaseUncertainty1D</a>
uncertainty for general information.
</p>

<h4>See also</h4>
<ul>
  <li>
    <a href=\"modelica://Credibility.Types.UniformTolerance\">UniformTolerance</a>
    for additional information.
  </li>
  <li>
    <a href=\"modelica://Credibility.Utilities.Table1DScalings.getTableLambdaByTolerance\">getTableLambdaByTolerance</a>
    for usability.
  </li>
</ul>
</html>"));
end UniformTolerance1D;
