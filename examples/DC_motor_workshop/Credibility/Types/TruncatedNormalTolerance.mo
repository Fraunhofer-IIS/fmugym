within Credibility.Types;
record TruncatedNormalTolerance "Uncertainty defined by truncated normal distribution with tolerance"
  extends TruncatedNormal(
    final kind = UncertaintyKind.TruncatedNormalTolerance,
    final lower = nominal - max(absTol,relTol*abs(nominal)),
    final upper = nominal + max(absTol,relTol*abs(nominal)),
    final mean = nominal,
    final stdDev = max(absTol,relTol*abs(nominal))/stdDevFactor);

  parameter Real nominal(final unit=unitValue) "Nominal value";
  parameter Real relTol=0.0
    "Limits: |value - nominal| <= max(absTol,relTol*|nominal|)";
  parameter Real absTol(final unit=unitValue)=0.0
    "Limits: |value - nominal| <= max(absTol,relTol*|nominal|)";
  parameter Real stdDevFactor=3
    "Standard deviation = max(absTol,relTol*|nominal|)/stdDevFactor"
    annotation(choices(
        choice=1 "1 (68.2%)",
        choice=2 "2 (95.4%)",
        choice=3 "3 (99.7%)",
        choice=4 "4 (99.99%)",
        choice=5 "5 (99.9999%)"));

  annotation (
    Documentation(info="<html>
<p>
A&nbsp;<em>truncated normally distributed</em> uncertainty. The minimum
and the maximum possible value of the uncertain scalar is described by
a&nbsp;<em>nominal value</em> and an <em>interval</em>.
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
    <em>relTol</em>: relative tolerance of limits with respect to nominal (default = 0.0).
  </li>
  <li>
    <em>absTol</em>: absolute tolerance of limits with respect to nominal (default = 0.0).
  </li>
</ul>
<p>
And, additionally,
</p>
<ul>
  <li>
    <em>stdDevFactor</em>: factor of tolerance to standard deviation of (non-truncated)
    normal distribution.
  </li>
</ul>

<p>
Consequently, the parameters of the underlaid
<a href=\"modelica://Credibility.Types.TruncatedNormal\">truncated normal
distribution</a>, are computed from the given parameterization in the following way:
</p>

<blockquote><pre>
tol = max(absTol, relTol * |nominal|)
lower = nominal &minus; tol
upper = nominal + tol
stdDev = tol/stdDevFactor
</pre></blockquote>

<p>
For example, the parameter&nbsp;<var>R</var> of a&nbsp;resistor has the nominal
value of 200&nbsp;&ohm;, the minimum possible value of 190&nbsp;&ohm; and the
maximum possible value 210&nbsp;&ohm;. 
These limits are considered being at 2*stdDev (standard deviation &sigma;) of the
<em>non-truncated</em> normal distribution, i.e. at a&nbsp;probability of 95.4&nbsp;%.
This yields the following parameters of truncated normal distribution:
</p>
<ul>
  <li>
    200&nbsp;&ohm; &plusmn; 5&nbsp;%, 2&sigma; &rArr; parameters
    <code>unitValue&nbsp;=\"Ohm\"</code>,
    <code>nominal&nbsp;= 200</code>,
    <code>relTol&nbsp;= 0.05</code> (and <code>absTol&nbsp;= 0</code>),
    <code>stdDevFactor&nbsp;= 2</code> or
  </li>
  <li>
    (200 &plusmn;&nbsp;10)&nbsp;&ohm;, 2&sigma; &rArr; parameters
    <code>unitValue&nbsp;=\"Ohm\"</code>,
    <code>nominal&nbsp;= 200</code>,
    <code>absTol&nbsp;= 10</code> (and <code>relTol&nbsp;= 0</code>),
    <code>stdDevFactor&nbsp;= 2</code>.
  </li>
</ul>
</html>"));
end TruncatedNormalTolerance;
