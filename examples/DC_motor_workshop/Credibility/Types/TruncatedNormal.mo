within Credibility.Types;
record TruncatedNormal "Uncertainty defined by truncated normal distribution"
  extends UndefinedUncertainty;
  extends BaseUncertainty(
    kind = UncertaintyKind.TruncatedNormal);
  parameter Real mean "Mean of (non-truncated) normal distribution";
  parameter Real stdDev "Standard deviation of (non-truncated) normal distribution";

  annotation (
    Documentation(info="<html>
<p>
A&nbsp;<em>truncated normally distributed</em> uncertainty limited by the
<em>minimum</em> and the <em>maximum</em> possible value of the uncertain scalar.
See also <a href=\"modelica://Credibility.Types.BaseUncertainty\">BaseUncertainty</a>
uncertainty for general information.
</p>
<p>
The additional parameters introduced for the truncated normal distribution are:
</p>
<ul>
  <li>
    <em>mean</em>: nominal value of the scalar is the mean value of non-truncated
    normal distribution.
  </li>
  <li>
    <em>stdDev</em>: standard deviation of non-truncated normal distribution.
  </li>
</ul>

<p>
For example, the parameter&nbsp;<var>R</var> of a&nbsp;resistor has the nominal
value of 200&nbsp;&ohm;, the minimum possible value 190&nbsp;&ohm;, the
maximum possible value 210&nbsp;&ohm; and the standard deviation 3&nbsp;&ohm;.
This yields the following parameters of truncated normal distribution:
</p>
<ul>
  <li>
    200&nbsp;&ohm; &plusmn; 5&nbsp;%, &sigma;&nbsp;= 3&nbsp;&ohm; &rArr; parameters
    <code>unitValue&nbsp;=\"Ohm\"</code>,
    <code>nominal&nbsp;= 200</code>,
    <code>relTol&nbsp;= 0.05</code> (and <code>absTol&nbsp;= 0</code>),
    <code>stdDev&nbsp;= 3</code> or
  </li>
  <li>
    (200 &plusmn;&nbsp;10)&nbsp;&ohm;, &sigma;&nbsp;= 3&nbsp;&ohm; &rArr; parameters
    <code>unitValue&nbsp;=\"Ohm\"</code>,
    <code>nominal&nbsp;= 200</code>,
    <code>absTol&nbsp;= 10</code> (and <code>relTol&nbsp;= 0</code>),
    <code>stdDev&nbsp;= 3</code>.
  </li>
</ul>

<p>
The Modelica Standard Library provides
<a href=\"modelica://Modelica.Math.Distributions.TruncatedNormal\">Modelica.Math.Distributions.TruncatedNormal</a>.
For more details of the normal distribution or of truncated distributions, see
Wikipedia &ndash;
<a href=\"http://en.wikipedia.org/wiki/Normal_distribution\">normal distribution</a>
or
<a href=\"http://en.wikipedia.org/wiki/Truncated_distribution\">truncated distribution</a>,
respectively.
</p>
</html>"));
end TruncatedNormal;
