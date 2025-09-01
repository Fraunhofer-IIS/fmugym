model dummy_geom_tol
  Modelica.Blocks.Interfaces.RealOutput output1 annotation(
    Placement(transformation(origin = {84, 42}, extent = {{-10, -10}, {10, 10}}), iconTransformation(extent = {{-10, -10}, {10, 10}})));
  Modelica.Blocks.Interfaces.RealOutput output2 annotation(
    Placement(transformation(origin = {84, -20}, extent = {{-10, -10}, {10, 10}}), iconTransformation(extent = {{-10, -10}, {10, 10}})));
  Modelica.Blocks.Interfaces.RealInput input2 annotation(
    Placement(transformation(origin = {-90, -20}, extent = {{-20, -20}, {20, 20}}), iconTransformation(origin = {-102, -26}, extent = {{-20, -20}, {20, 20}})));
  Modelica.Blocks.Interfaces.RealInput input1 annotation(
    Placement(visible = true, transformation(origin = {-90, 42}, extent = {{-20, -20}, {20, 20}}, rotation = 0)));
  Modelica.Blocks.Math.Add add(k1 = -1)  annotation(
    Placement(transformation(origin = {8, -20}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Blocks.Math.Gain gain(k = 2.0)  annotation(
    Placement(transformation(origin = {-42, 42}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Blocks.Math.Gain gain1(k = 3.0)  annotation(
    Placement(transformation(origin = {-42, -20}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Blocks.Math.Sin sin annotation(
    Placement(transformation(origin = {46, -20}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Blocks.Math.Atan atan annotation(
    Placement(transformation(origin = {46, 42}, extent = {{-10, -10}, {10, 10}})));
equation
  connect(atan.y, output1) annotation(
    Line(points = {{58, 42}, {84, 42}}, color = {0, 0, 127}));
  connect(sin.y, output2) annotation(
    Line(points = {{57, -20}, {84, -20}}, color = {0, 0, 127}));
  connect(add.y, sin.u) annotation(
    Line(points = {{19, -20}, {34, -20}}, color = {0, 0, 127}));
  connect(input1, gain.u) annotation(
    Line(points = {{-90, 42}, {-54, 42}}, color = {0, 0, 127}));
  connect(input2, gain1.u) annotation(
    Line(points = {{-90, -20}, {-54, -20}}, color = {0, 0, 127}));
  connect(gain.y, atan.u) annotation(
    Line(points = {{-30, 42}, {34, 42}}, color = {0, 0, 127}));
  connect(gain.y, add.u1) annotation(
    Line(points = {{-30, 42}, {-20, 42}, {-20, -14}, {-4, -14}}, color = {0, 0, 127}));
  connect(gain1.y, add.u2) annotation(
    Line(points = {{-30, -20}, {-16, -20}, {-16, -26}, {-4, -26}}, color = {0, 0, 127}));
  annotation(
    uses(Modelica(version = "4.0.0")),
    Diagram(coordinateSystem(extent = {{-200, -200}, {200, 200}})),
    Icon(coordinateSystem(extent = {{-200, -200}, {200, 200}})),
    version = "");
end dummy_geom_tol;