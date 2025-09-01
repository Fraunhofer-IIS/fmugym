within Credibility.Examples.SimpleControlledDriveNonlinear;
partial record PartialData "Base record of credible parameters of simple control drive"
  extends Modelica.Icons.Record;

  parameter String variantName= "PartialData" "Name of identification variant";

  parameter Credibility.Scalar Ja(redeclare parameter Modelica.Units.SI.Inertia value) "Inertia of load";

  parameter Credibility.Scalar Jm(redeclare parameter Modelica.Units.SI.Inertia value) "Inertia of drive";

  parameter Credibility.Scalar ratio "Gearbox transmission ratio (flange_a.phi/flange_b.phi)";

  parameter Credibility.Scalar damping(value(unit="N.m.s/rad")) "Damper constant of drive compliance";

  parameter Credibility.Table1D stiffness "Table for nonlinear spring stiffness in N.m/rad of drive compliance";

  annotation (
    Icon(coordinateSystem(preserveAspectRatio=false),
      graphics={
        Text(
          extent={{-150,-8},{150,-38}},
          textColor={0,140,72},
          textString="%variantName",
          textStyle={TextStyle.Bold})}),
    Diagram(coordinateSystem(preserveAspectRatio=false)),
    Documentation(info="<html>
<p>
A&nbsp;partial record containing all credible parameters needed for
a&nbsp;<a href=\"modelica://Credibility.Examples.SimpleControlledDriveNonlinear.PartialDrive\">simple drive</a>.
The parameters have only dummy values. A&nbsp;particular parameterization
shall be done in a&nbsp;record which extends this one, see e.g.
<a href=\"modelica://Credibility.Examples.SimpleControlledDriveNonlinear.DataVariant001\">DataVariant001</a>.
</p>
</html>"));
end PartialData;
