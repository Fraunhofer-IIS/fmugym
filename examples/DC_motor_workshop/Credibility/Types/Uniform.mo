within Credibility.Types;
record Uniform "Uncertainty defined by uniform distribution"
  extends UndefinedUncertainty;
  extends BaseUncertainty(
    kind = UncertaintyKind.Uniform);

  annotation (
    Documentation(info="<html>
<p>
An uncertainty describing possible value of the uncertain scalar by a&nbsp;<em>uniform
distribution</em>, i.e. the probability of value outcome is equal between the
<em>minimum</em> and the <em>maximum</em> limits.
See also <a href=\"modelica://Credibility.Types.BaseUncertainty\">BaseUncertainty</a>
uncertainty for general information.
</p>
<p>
The <em>parameter value</em> is here considered being equal to the <em>nominal
value</em> which is used in an optimization run that determines improved control
parameters or the nominal value used during a&nbsp;Monte Carlo simulation to
propagate uncertainties to output variables.
</p>

<p>
The Modelica Standard Library provides
<a href=\"modelica://Modelica.Math.Distributions.Uniform\">Modelica.Math.Distributions.Uniform</a>.
For more details of the uniform distribution, see
Wikipedia &ndash;
<a href=\"https://en.wikipedia.org/wiki/Continuous_uniform_distribution\">continuous uniform distribution</a>.
</p>
</html>"));
end Uniform;
