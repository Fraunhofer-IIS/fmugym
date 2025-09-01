within Credibility.Types;
record Interval1D "Interval uncertainty of lambda for table1D (epistemic uncertainty)"
  extends BaseUncertainty1D(
    kind = UncertaintyKind1D.Interval1D);

  constant Boolean withTolerance=false "= true, if uncertainty defined by tolerance" annotation (HideResult=true);
  constant Real absTol=0.0 "Dummy" annotation (HideResult=true);
  constant Real relTol=0.0 "Dummy" annotation (HideResult=true);
  parameter Real table[:,4] = zeros(1,4)
    "Table matrix (columns: 1 = input/grid; 2 = nominal values; 3 = lower limits; 4 = upper limits)";
  annotation (
    Documentation(info="<html>
<p>
An <em>epistemic</em> uncertainty of 1d table, i.e. the uncertainty <em>due to lack
of knowledge</em> &ndash; e.g., on the side of model, analysis or experiment.
The uncertainty is described only by vectors of the <em>minimum</em> and the <em>maximum</em>
possible values of the uncertain table.
See also <a href=\"modelica://Credibility.Types.BaseUncertainty1D\">BaseUncertainty1D</a>
uncertainty for general information.
</p>
<p>
The <em>nominal</em>, <em>minimum</em> and <em>maximum</em> vectors are all
collected in <code>table</code>.
To use this uncertainty in an optimization run that determines improved control
parameters or during a&nbsp;Monte Carlo simulation to propagate uncertainties to
output variables, the uncertain parameter <code>lambda</code> is introduced.
It is aimed to enable convex scaling between lower limits (for <code>lambda&nbsp;= -1</code>),
nominal values (<code>lambda&nbsp;= 0</code>) and upper limits (<code>lambda&nbsp;= 1</code>).
</p>
</html>"));
end Interval1D;
