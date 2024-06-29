model dummy_for_FMU
  Modelica.Blocks.Continuous.FirstOrder firstOrder annotation(
    Placement(transformation(origin = {-4, 42}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Blocks.Continuous.FirstOrder firstOrder1 annotation(
    Placement(transformation(origin = {-4, -20}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Blocks.Interfaces.RealOutput output1 annotation(
    Placement(transformation(origin = {84, 42}, extent = {{-10, -10}, {10, 10}}), iconTransformation(extent = {{-10, -10}, {10, 10}})));
  Modelica.Blocks.Interfaces.RealOutput output2 annotation(
    Placement(transformation(origin = {84, -20}, extent = {{-10, -10}, {10, 10}}), iconTransformation(extent = {{-10, -10}, {10, 10}})));
  Modelica.Blocks.Interfaces.RealInput input2 annotation(
    Placement(transformation(origin = {-90, -20}, extent = {{-20, -20}, {20, 20}}), iconTransformation(origin = {-102, -26}, extent = {{-20, -20}, {20, 20}})));
  Modelica.Blocks.Interfaces.RealInput input1 annotation(
    Placement(visible = true, transformation(origin = {-90, 42}, extent = {{-20, -20}, {20, 20}}, rotation = 0)));
equation
  connect(firstOrder1.y, output2) annotation(
    Line(points = {{8, -20}, {84, -20}}, color = {0, 0, 127}));
  connect(input2, firstOrder1.u) annotation(
    Line(points = {{-90, -20}, {-16, -20}}, color = {0, 0, 127}));
  connect(input1, firstOrder.u) annotation(
    Line(points = {{-90, 42}, {-16, 42}}, color = {0, 0, 127}));
  connect(firstOrder.y, output1) annotation(
    Line(points = {{8, 42}, {84, 42}}, color = {0, 0, 127}));
  annotation(
    uses(Modelica(version = "4.0.0")),
    Diagram(coordinateSystem(extent = {{-200, -200}, {200, 200}})),
    Icon(coordinateSystem(extent = {{-200, -200}, {200, 200}})),
    version = "");
end dummy_for_FMU;