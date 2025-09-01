within Credibility.Types;
record Uniform1D "Uniform distribution of lambda for table1D"
  extends Interval1D(
    final kind=Types.UncertaintyKind1D.Uniform1D);

  annotation (Documentation(info="<html>
<p>
An uncertainty describing possible values of the uncertain 1d table by a&nbsp;<em>uniform
distribution</em>, i.e. the probability of values&apos; outcome is equal between the
<em>minimum</em> and the <em>maximum</em> limits.
See also <a href=\"modelica://Credibility.Types.BaseUncertainty1D\">BaseUncertainty1D</a>
uncertainty for general information.
</p>


<h4>See also</h4>
<ul>
  <li>
    <a href=\"modelica://Credibility.Types.Uniform\">Uniform</a>
    for additional information.
  </li>
</ul>
</html>"));
end Uniform1D;
