within Credibility.Types;
record TruncatedNormal1D "Truncated normal distribution of lambda for table1D"
  extends Interval1D(
    kind = UncertaintyKind1D.TruncatedNormal1D);

  parameter Real stdDev "Standard deviation of (non-truncated) normal distribution";

  annotation (Documentation(info="<html>
<p>
A&nbsp;<em>truncated normally distributed</em> uncertainty, limited by the
<em>minimum</em> and the <em>maximum</em> possible values of 1d table.
See also <a href=\"modelica://Credibility.Types.BaseUncertainty1D\">BaseUncertainty1D</a>
uncertainty for general information.
</p>
<p>
The additional parameter introduced for this truncated normal distribution is:
</p>
<ul>
  <li>
    <em>stdDev</em>: standard deviation of non-truncated normal distribution.
  </li>
</ul>


<h4>See also</h4>
<ul>
  <li>
    <a href=\"modelica://Credibility.Types.TruncatedNormal\">TruncatedNormal</a>
  </li>
</ul>
</html>"));
end TruncatedNormal1D;
