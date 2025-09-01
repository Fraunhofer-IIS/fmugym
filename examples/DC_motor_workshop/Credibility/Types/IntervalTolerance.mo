within Credibility.Types;
record IntervalTolerance "Uncertainty defined by interval (epistemic uncertainty) with tolerance"
  extends Interval(
    final kind = UncertaintyKind.IntervalTolerance,
    final lower = nominal - max(absTol,relTol*abs(nominal)),
    final upper = nominal + max(absTol,relTol*abs(nominal)));

  parameter Real nominal(final unit=unitValue) "Nominal value";
  parameter Real relTol=0.0
    "Limits: |value - nominal| <= max(absTol, relTol*|nominal|)";
  parameter Real absTol(final unit=unitValue) = 0.0
    "Limits: |value - nominal| <= max(absTol, relTol*|nominal|)";

  annotation (
    Documentation(info="<html>
<p>
An <em>epistemic</em> uncertainty, i.e. the uncertainty <em>due to lack
of knowledge</em> &ndash; e.g., on the side of model, analysis or experiment.
The uncertainty is described by a&nbsp;<em>nominal value</em> and an
<em>interval</em>.
See also <a href=\"modelica://Credibility.Types.BaseUncertainty\">BaseUncertainty</a>
uncertainty for general information.
</p>
<p>
Often, it is inconvenient to provide absolute ranges of a&nbsp;value and,
instead, relative or absolute deviations are more practical. In the eFMI standard
(<a href=\"https://emphysis.github.io/pages/downloads/efmi_specification_1.0.0-alpha.4.zip\">Functional
Mock-up Interface for embedded systems</a>), for example, tolerances
for reference results are defined in a&nbsp;similar way as tolerances for
numerical integration algorithms. Due to its generality, this description
form of eFMI is used here as well:
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
<a href=\"modelica://Credibility.Types.Interval\">interval
distribution</a> can be computed in
the following way:
</p>

<blockquote><pre>
tol = max(absTol, relTol * |nominal|),
lower = nominal &minus; tol,
upper = nominal + tol.
</pre></blockquote>

<p>
For example, the parameter&nbsp;<var>R</var> of a&nbsp;resistor has the nominal
value of 200&nbsp;&ohm;, the minimum possible value of 190&nbsp;&ohm; and the
maximum possible value 210&nbsp;&ohm;. Then, the following tolerances can be
provided:
</p>
<ul>
  <li>
    200&nbsp;&ohm; &plusmn; 5&nbsp;% &rArr; parameters
    <code>unitValue&nbsp;=\"Ohm\"</code>,
    <code>nominal&nbsp;= 200</code>,
    <code>relTol&nbsp;= 0.05</code> (and <code>absTol&nbsp;= 0</code>) or
  </li>
  <li>
    (200 &plusmn;&nbsp;10)&nbsp;&ohm; &rArr; parameters
    <code>unitValue&nbsp;=\"Ohm\"</code>,
    <code>nominal&nbsp;= 200</code>,
    <code>absTol&nbsp;= 10</code> (and <code>relTol&nbsp;= 0</code>).
  </li>
</ul>
<p>
Typically, either a&nbsp;description with relTol or with absTol is used, but
not a&nbsp;combination of both. On the contrary, there is an important use case
where the description with both relTol and absTol is useful:
If reference results are provided (e.g. the results of a&nbsp;simulation or an
experiment are required to be within some band around a&nbsp;provided reference
solution) then a&nbsp;definition with relTol is often practical together with
absTol as band for small values of the reference around zero, where relTol
makes no sense.
</p>
</html>"));
end IntervalTolerance;
