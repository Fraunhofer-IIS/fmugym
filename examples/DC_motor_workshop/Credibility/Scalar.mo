within Credibility;
record Scalar "Record collecting credibility information for a scalar Real parameter"
  extends Credibility.Icons.CredibleBase;

  replaceable parameter Real value = 0 "Value of parameter";

  parameter Types.Traceability traceability "Traceability";

  replaceable parameter Types.UndefinedUncertainty uncertainty
    constrainedby Types.UndefinedUncertainty "Uncertainty"
    annotation (choicesAllMatching=true);
  replaceable parameter Types.UndefinedCalibration calibration
    constrainedby Types.UndefinedCalibration "Calibration setup"
    annotation (choicesAllMatching=true);

  annotation (
    Icon(graphics={
        Text(
          extent={{-50,-10},{100,-60}},
          textColor={0,140,72},
          textString="scalar")}),
    Documentation(info="<html>
<p>
This record collects credibility information for a&nbsp;<em>scalar</em>
parameter. In particular, it contains information regarding
</p>
<ul>
  <li>
    the Real value of the scalar parameter,
  </li>
  <li>
    traceability,
  </li>
  <li>
    uncertainty and
  </li>
  <li>
    calibration.
  </li>
</ul>
</html>"));
end Scalar;
