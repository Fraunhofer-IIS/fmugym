model NonlinearMIMO
  Modelica.Blocks.Continuous.FirstOrder firstOrder(k = 5, T = 1)  annotation(
    Placement(transformation(origin = {66, 80}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Blocks.Continuous.FirstOrder firstOrder1(k = 4, T = 3)  annotation(
    Placement(transformation(origin = {68, -20}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Blocks.Interfaces.RealOutput output1 annotation(
    Placement(transformation(origin = {170, 80}, extent = {{-10, -10}, {10, 10}}), iconTransformation(extent = {{-10, -10}, {10, 10}})));
  Modelica.Blocks.Interfaces.RealOutput output2 annotation(
    Placement(transformation(origin = {170, -20}, extent = {{-10, -10}, {10, 10}}), iconTransformation(extent = {{-10, -10}, {10, 10}})));
  Modelica.Blocks.Interfaces.RealInput input2 annotation(
    Placement(transformation(origin = {-144, -20}, extent = {{-20, -20}, {20, 20}}), iconTransformation(origin = {-102, -26}, extent = {{-20, -20}, {20, 20}})));
  Modelica.Blocks.Interfaces.RealInput input1 annotation(
    Placement(transformation(origin = {-144, 80}, extent = {{-20, -20}, {20, 20}}), iconTransformation(origin = {-90, 42}, extent = {{-20, -20}, {20, 20}})));
  Modelica.Blocks.Math.Tanh tanh annotation(
    Placement(transformation(origin = {-48, 80}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Blocks.Nonlinear.DeadZone deadZone(uMax = 2)  annotation(
    Placement(transformation(origin = {-42, -20}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Blocks.Math.Add add annotation(
    Placement(transformation(origin = {12, 28}, extent = {{-10, -10}, {10, 10}})));
equation
  connect(firstOrder.y, output1) annotation(
    Line(points = {{77, 80}, {170, 80}}, color = {0, 0, 127}));
  connect(firstOrder1.y, output2) annotation(
    Line(points = {{79, -20}, {170, -20}}, color = {0, 0, 127}));
  connect(deadZone.u, input2) annotation(
    Line(points = {{-54, -20}, {-144, -20}}, color = {0, 0, 127}));
  connect( input1, tanh.u) annotation(
    Line(points = {{-144, 80}, {-60, 80}}, color = {0, 0, 127}));
  connect(tanh.y, firstOrder.u) annotation(
    Line(points = {{-36, 80}, {54, 80}}, color = {0, 0, 127}));
  connect(deadZone.y, add.u2) annotation(
    Line(points = {{-30, -20}, {-20, -20}, {-20, 22}, {0, 22}}, color = {0, 0, 127}));
  connect(tanh.y, add.u1) annotation(
    Line(points = {{-36, 80}, {-12, 80}, {-12, 34}, {0, 34}}, color = {0, 0, 127}));
  connect(add.y, firstOrder1.u) annotation(
    Line(points = {{24, 28}, {40, 28}, {40, -20}, {56, -20}}, color = {0, 0, 127}));
  annotation(
    uses(Modelica(version = "4.0.0")));
end NonlinearMIMO;