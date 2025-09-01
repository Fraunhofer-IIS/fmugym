within Credibility.Types;
record UniformTolerance "Uncertainty defined by uniform distribution with tolerance"
  extends Uniform(
     final kind = UncertaintyKind.UniformTolerance,
     final lower = nominal - max(absTol,relTol*abs(nominal)),
     final upper = nominal + max(absTol,relTol*abs(nominal)));

  parameter Real nominal(final unit=unitValue) "Nominal value";
  parameter Real relTol=0.0
    "Limits: |value - nominal| <= max(absTol,relTol*|nominal|)";
  parameter Real absTol(final unit=unitValue)=0.0
    "Limits: |value - nominal| <= max(absTol,relTol*|nominal|)";

  annotation (
    Documentation(info="<html>
<p>
An uncertainty describing possible value of the uncertain scalar by a&nbsp;<em>uniform
distribution</em>, i.e. the probability of value outcome is equal between the
limits which are given by a&nbsp;<em>nominal value</em> and an
<em>interval</em>.
See also <a href=\"modelica://Credibility.Types.BaseUncertainty\">BaseUncertainty</a>
uncertainty for general information.
</p>
<p>
Similarly to
<a href=\"modelica://Credibility.Types.IntervalTolerance\">IntervalTolerance</a>,
this description utilizes:
</p>
<ul>
  <li>
    <em>nominal</em>: nominal value of the scalar. Typically, the nominal value
    is used as default for the actual parameter value.
  </li>
  <li>
    <em>relTol</em>: relative tolerance of limits with respect to nominal
    (default&nbsp;=&nbsp;0.0).
  </li>
  <li>
    <em>absTol</em>: absolute tolerance of limits with respect to nominal
    (default&nbsp;=&nbsp;0.0).
  </li>
</ul>
<p>
Utilizing this information, the lower and upper values of the underlaid
<a href=\"modelica://Credibility.Types.Uniform\">uniform distribution</a>
can be computed in the following way:
</p>

<blockquote><pre>
tol = max(absTol, relTol * |nominal|),
lower = nominal &minus; tol,
upper = nominal + tol.
</pre></blockquote>
</html>"));
end UniformTolerance;
