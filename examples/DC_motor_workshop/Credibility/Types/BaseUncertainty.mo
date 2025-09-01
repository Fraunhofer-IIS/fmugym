within Credibility.Types;
record BaseUncertainty "Basic uncertainty definitions"
  constant UncertaintyKind kind "Kind of uncertainty";
  parameter SourceType source = SourceType.Unknown "Source of uncertainty";
  parameter String info = "" "Information about the source";
  parameter String reference = "" "Reference of the source";
  constant  String unitValue "Unit of value";
  parameter Real lower(final unit=unitValue) "Minimum possible value (lower <= value <= upper)";
  parameter Real upper(final unit=unitValue) "Maximum possible value (lower <= value <= upper)";
  annotation (
    Icon(graphics={
        Text(
          textColor={0,0,255},
          extent={{-150,60},{150,100}},
          textString="%name"),
        Rectangle(
          extent={{-100,48},{100,-100}},
          lineColor={0,0,0},
          fillColor={255,255,255},
          fillPattern=FillPattern.Solid),
        Line(points={{-76,30},{80,30}}, color={0,140,72}),
        Line(points={{-76,-80},{80,-80}}, color={0,140,72}),
        Ellipse(
          extent={{-64,-44},{-40,-68}},
          lineColor={0,140,72},
          fillColor={0,140,72},
          fillPattern=FillPattern.Solid),
        Ellipse(
          extent={{-12,16},{12,-8}},
          lineColor={0,140,72},
          fillColor={0,140,72},
          fillPattern=FillPattern.Solid),
        Ellipse(
          extent={{46,-30},{70,-54}},
          lineColor={0,140,72},
          fillColor={0,140,72},
          fillPattern=FillPattern.Solid)}),
    Documentation(info="<html>
<p>
A&nbsp;common uncertainty setup for a&nbsp;<em>scalar real parameter</em>.
Besides <em>source</em>, <em>info</em> and <em>reference</em>,
detailed in
<a href=\"modelica://Credibility.UsersGuide.ParameterCredibility.UncertaintyInfo\">User&apos;s Guide</a>,
inherent limits of a&nbsp;physical quantity (i.e., upper and lower
limits of a&nbsp;scalar value) are defined here, independent of
the kind of the mathematical description of the uncertainty.
A&nbsp;particular mathematical description is, then, provided with the derived
uncertainty types, see e.g. 
<a href=\"modelica://Credibility.Types.Interval\">Interval</a>
or
<a href=\"modelica://Credibility.Types.TruncatedNormalTolerance\">TruncatedNormalTolerance</a>.
</p>
</html>"));
end BaseUncertainty;
