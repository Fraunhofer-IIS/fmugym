within Credibility.Types;
record BaseUncertainty1D "Basic uncertainty definitions of 1D tables"
  constant UncertaintyKind1D kind "Kind of 1D uncertainty";
  parameter SourceType source = SourceType.Unknown "Source of uncertainty";
  parameter String info = "" "Information about the source";
  parameter String reference = "" "Reference of the source";
  parameter Real lambda(min=-1, max=1) = 0 "Convex scaling between -1 (= lower limits), 0 (= nominal) and 1 (= upper limits)";

  annotation (Icon(graphics={
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
A&nbsp;common uncertainty setup for a&nbsp;<em>real 1d&nbsp;table parameter</em>.
Besides <em>source</em>, <em>info</em> and <em>reference</em>,
detailed in
<a href=\"modelica://Credibility.UsersGuide.ParameterCredibility.UncertaintyInfo\">User&apos;s Guide</a>,
inherent limits of a&nbsp;physical quantity (i.e., upper and lower
limits of table values) are defined here, independent of
the kind of the mathematical description of the uncertainty.
Additionally, the uncertain parameter <code>lambda</code> is aimed for scaling
between lower limits (for <code>lambda&nbsp;= -1</code>), nominal values
(<code>lambda&nbsp;= 0</code>) and upper limits (<code>lambda&nbsp;= 1</code>), see
<a href=\"modelica://Credibility.UsersGuide.ParameterCredibility.UncertaintyInfo\">User&apos;s Guide</a>.
</p>
<p>
A&nbsp;particular mathematical description is, then, provided with the derived
uncertainty types, see e.g. 
<a href=\"modelica://Credibility.Types.Interval1D\">Interval1D</a>
or
<a href=\"modelica://Credibility.Types.TruncatedNormalTolerance1D\">TruncatedNormalTolerance1D</a>.
</p>
</html>"));
end BaseUncertainty1D;
