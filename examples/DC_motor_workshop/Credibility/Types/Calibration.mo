within Credibility.Types;
record Calibration "Calibration setup for scalar parameter"
  extends UndefinedCalibration;
  extends Modelica.Icons.Record;
  constant String unitValue "Unit of value";
  parameter Real start(final unit=unitValue) = 0.0 "Start value for calibration";
  parameter Real lower(final unit=unitValue) = 0.0 "Minimum value used for calibration";
  parameter Real upper(final unit=unitValue) = 0.0 "Maximum value used for calibration";
  parameter String setup = "" "URL of calibration setup script";

  annotation (
    Documentation(info="<html>
<p>
A&nbsp;common calibration setup for a&nbsp;scalar real parameter. See also
<a href=\"modelica://Credibility.UsersGuide.ParameterCredibility.CalibrationInfo\">User&apos;s Guide</a>
for more info.
</p>
</html>"));
end Calibration;
