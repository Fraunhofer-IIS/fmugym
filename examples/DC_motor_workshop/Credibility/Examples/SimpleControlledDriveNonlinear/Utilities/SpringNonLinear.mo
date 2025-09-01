within Credibility.Examples.SimpleControlledDriveNonlinear.Utilities;
model SpringNonLinear "Nonlinear rotational spring"
  extends Modelica.Mechanics.Rotational.Interfaces.PartialCompliant;

  parameter Modelica.Units.SI.Angle phi_rel0=0 "Unstretched spring angle";

  parameter Boolean tableOnFile=true
    "True, if stiffness table is defined on file"
    annotation (Dialog(group="Spring characteristics"));
  parameter Real table[:, :]=[0, 0; 1, 0]
    "Spring stiffness [N.m.rad-1] (col. 2) vs. deflection angle delta_phi_rel [rad] (col. 1)"
    annotation (Dialog(group="Spring characteristics", enable=not tableOnFile));
  parameter String fileName="NoName" "File name where table is stored"
    annotation (Dialog(group="Spring characteristics", enable=tableOnFile));
  parameter String tableName="NoName" "Stiffness table name"
    annotation (Dialog(group="Spring characteristics", enable=tableOnFile));
  parameter Real k=1 "Coefficient for scaling of torque"
    annotation (Dialog(group="Spring characteristics"));

  replaceable Modelica.Blocks.Tables.CombiTable1Ds c_nonlin(
    final tableOnFile=tableOnFile,
    final table=table,
    final columns={2},
    final tableName=tableName,
    final fileName=fileName,
    smoothness=Modelica.Blocks.Types.Smoothness.LinearSegments)
    constrainedby Modelica.Blocks.Interfaces.SIMO
    "Force look-up table" annotation (Placement(
        transformation(extent={{-10,-10},{10,10}}, rotation=0)));

equation
  c_nonlin.u = phi_rel - phi_rel0;
  tau = k*c_nonlin.y[1]*(phi_rel - phi_rel0);

  annotation (
    Documentation(info="<html>
<p>
A&nbsp;1D&nbsp;rotational spring element with non-linear characteristic.
The torque is defined according to:
</p>
<blockquote><pre>
tau = k * c_nonlin(delta_phi_rel),
</pre></blockquote>
<p>
where
</p>
<blockquote><pre>
tau ............. resulting torque,
c_nonlin ........ non-linear stiffness table,
delta_phi_rel ... = phi_rel - phi_rel0,
k ............... scaling coefficient.
</pre></blockquote>
</html>"),
    Icon(
      coordinateSystem(
        preserveAspectRatio=false,
        extent={{-100,-100},{100,100}},
        grid={2,2}),
      graphics={
        Line(
          points={{-100,0},{-60,0},{-56,-28},{-50,30},{-44,-28},{-38,30},{-30,
              -28},{-20,30},{-6,-28},{14,30},{34,-28},{46,30},{60,0},{100,0}},
          color={0,0,0}),
        Rectangle(
          extent={{-40,20},{40,-20}},
          lineColor={0,0,0},
          fillColor={255,255,255},
          fillPattern=FillPattern.Solid),
        Text(
          extent={{-100,80},{100,40}},
          textString="%name",
          textColor={0,0,255}),
        Line(
          points={{-34,14},{-34,-14},{34,-14}},
          color={0,0,255}),
        Line(
          points={{-34,-14},{-30,-6},{-14,4},{14,8},{28,8}},
          color={0,0,255}),
        Text(
          extent={{-250,-40},{250,-80}},
          textString="table=%table",
          textColor={0,0,0})}),
    Diagram(
      coordinateSystem(
        preserveAspectRatio=false,
        extent={{-100,-100},{100,100}},
        grid={2,2}),
      graphics={
        Line(
          points={{-100,0},{-100,83}},
          color={128,128,128}),
        Line(
          points={{-100,80},{100,80}},
          color={128,128,128}),
        Text(
          extent={{-20,56},{20,81}},
          textColor={128,128,128},
          textString="phi_rel"),
        Polygon(
          points={{90,83},{100,80},{90,77},{90,83}},
          lineColor={128,128,128},
          fillColor={128,128,128},
          fillPattern=FillPattern.Solid),
        Line(
          points={{100,0},{100,83}},
          color={128,128,128}),
        Text(
          extent={{13,-63},{81,-76}},
          textColor={128,128,128},
          textString="translation axis")}));
end SpringNonLinear;
