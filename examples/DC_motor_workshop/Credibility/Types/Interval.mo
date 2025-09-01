within Credibility.Types;
record Interval "Uncertainty defined by interval (epistemic uncertainty)"
  extends UndefinedUncertainty;
  extends BaseUncertainty(
    kind = UncertaintyKind.Interval);

  annotation (
    Documentation(info="<html>
<p>
An <em>epistemic</em> uncertainty, i.e. the uncertainty <em>due to lack
of knowledge</em> &ndash; e.g., on the side of model, analysis or experiment.
The uncertainty is described only by the <em>minimum</em> and the <em>maximum</em>
possible value of the uncertain scalar.
See also <a href=\"modelica://Credibility.Types.BaseUncertainty\">BaseUncertainty</a>
uncertainty for general information.
</p>
<p>
The <em>parameter value</em> is here considered being equal to the <em>nominal
value</em> which is used in an optimization run that determines improved control
parameters or the nominal value used during a&nbsp;Monte Carlo simulation to
propagate uncertainties to output variables.
</p>
</html>"));
end Interval;
