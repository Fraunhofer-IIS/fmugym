within Credibility.Types;
record Traceability "Traceability setup"
  parameter SourceType source = SourceType.Unknown "Source of value";
  parameter String info = "Based on measurements on test bed B in project XYZ." "Information about the source of the value and its uncertainty information";
  parameter String reference = "modelica://ProjectXYZData/Drives/Variant002/ReportXYZ_B17.pdf" "URI of the source (report, data sheet, paper, ...)";

  annotation (
    Icon(
      coordinateSystem(preserveAspectRatio=false)),
    Diagram(
      coordinateSystem(preserveAspectRatio=false)),
    Documentation(info="<html>
<p>
A&nbsp;common traceability setup. See
<a href=\"modelica://Credibility.UsersGuide.ParameterCredibility.TraceabilityInfo\">User&apos;s Guide</a>
for more info.
</p>
</html>"));
end Traceability;
