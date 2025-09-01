within Credibility.Examples.SimpleControlledDriveNonlinear;
model DC_Motor_Workshop
extends Modelica.Icons.Example;
  extends Credibility.Examples.SimpleControlledDriveNonlinear.PartialDrive(redeclare SimpleControlledDriveNonlinear.DataVariant001 data, inertiaLoad(a(fixed = true, start = 0)), damper(d=40),     spring(k = 1));
  parameter Units.SI.RotationalDampingConstant d_exact = 100 "Exact value of damping constant used for virtual measurement";
  Modelica.Blocks.Continuous.LimPID speedController(k = 100, Ti = 0.1, Ni = 0.1, initType = Modelica.Blocks.Types.Init.SteadyState, controllerType = Modelica.Blocks.Types.SimpleController.PI, limiter(u(start = 0)), Td = 0.1, yMax = 12) "k, Ti, Ni: defined from control development, to be tested on control robustness ; yMax, yMin: from system spec; Init defined by optimization scenario spec" annotation (
    Placement(transformation(origin={-44,2},    extent = {{-90, -10}, {-70, 10}})));
  Modelica.Blocks.Sources.SineVariableFrequencyAndAmplitude sine(useConstantAmplitude = true,
    constantAmplitude=0.01,                                                                                             useConstantFrequency = false) annotation (
    Placement(transformation(extent = {{-50, 50}, {-30, 70}})));
  Modelica.Blocks.Sources.Ramp ramp(height = 30, duration = 4, offset = 1) annotation (
    Placement(transformation(extent = {{-90, 50}, {-70, 70}})));
  Modelica.Mechanics.Rotational.Sensors.SpeedSensor speedSensorLoad annotation (
    Placement(transformation(extent = {{10, -10}, {-10, 10}}, rotation = 90, origin = {50, -40})));
  Modelica.Blocks.Interfaces.RealOutput out_motor_speed "Connector of Real output signal" annotation (
    Placement(transformation(origin = {-100, 12}, extent = {{100, -70}, {120, -50}}), iconTransformation(extent = {{110, -60}, {110, -60}})));
  Modelica.Blocks.Interfaces.RealOutput out_load_speed "Connector of Real output signal" annotation (
    Placement(transformation(origin = {-22, 36}, extent = {{100, -100}, {120, -80}}), iconTransformation(extent = {{110, -90}, {110, -90}})));
  Modelica.Blocks.Math.Add add(k2 = -1) annotation (
    Placement(transformation(origin = {-144, -40}, extent = {{-8, -8}, {8, 8}})));
  Modelica.Blocks.Interfaces.RealOutput out_control_err "Connector of Real output signal" annotation (
    Placement(transformation(origin = {-225, 20}, extent = {{100, -70}, {120, -50}}), iconTransformation(origin = {-247, -50}, extent = {{110, -60}, {110, -60}})));
  Modelica.Blocks.Math.Add add3(y(start=0))
                                annotation (
    Placement(transformation(origin={-99,4.5},  extent={{-5,-5.5},{5,5.5}})));
  Modelica.Blocks.Interfaces.RealInput in_torque annotation (
    Placement(transformation(origin={-153,27},    extent = {{-7, -7}, {7, 7}}), iconTransformation(origin = {-146, 62}, extent = {{-20, -20}, {20, 20}})));
  Modelica.Mechanics.Rotational.Sources.TorqueStep torqueStep(
    stepTorque=5.0,                                                           offsetTorque = 5.0, startTime = 0.5)  annotation (
    Placement(transformation(origin = {100, 0}, extent = {{10, -10}, {-10, 10}})));
  Modelica.Blocks.Continuous.FirstOrder firstOrder(
    k=1,
    T=0.1,
    y_start=0)
    annotation (Placement(transformation(extent={{-158,-6},{-142,10}})));
  Modelica.Blocks.Nonlinear.Limiter limiter(uMax=12, uMin=-12)
    annotation (Placement(transformation(extent={{-84,-4},{-70,10}})));
initial equation

equation
  connect(ramp.y, sine.f) annotation (
    Line(points = {{-69, 60}, {-60, 60}, {-60, 54}, {-52, 54}}, color = {0, 0, 127}));
  connect(inertiaLoad.flange_a, speedSensorLoad.flange) annotation (
    Line(points = {{58, 0}, {50, 0}, {50, -30}}, color = {0, 0, 0}));
  connect(add.y, out_control_err) annotation (
    Line(points = {{-135.2, -40}, {-115, -40}}, color = {0, 0, 127}));
  connect(add.u1, sine.y) annotation (
    Line(points = {{-153.6, -35.2}, {-176, -35.2}, {-176, 36}, {-20, 36}, {-20, 60}, {-29, 60}}, color = {0, 0, 127}));
  connect(in_torque, add3.u1) annotation (
    Line(points={{-153,27},{-142,27},{-142,28},{-112,28},{-112,7.8},{-105,7.8}},
                                                                  color = {0, 0, 127}));
  connect(torqueStep.flange, inertiaLoad.flange_b) annotation (
    Line(points = {{90, 0}, {78, 0}}));
  connect(speedSensor.w, out_motor_speed) annotation (
    Line(points = {{-9, -33}, {-10, -33}, {-10, -48}, {10, -48}}, color = {0, 0, 127}));
  connect(speedSensorLoad.w, speedController.u_m) annotation (
    Line(points={{50,-51},{50,-70},{-102,-70},{-102,-22},{-124,-22},{-124,-10}},            color = {0, 0, 127}));
  connect(speedSensorLoad.w, out_load_speed) annotation (
    Line(points = {{50, -51}, {50, -54}, {88, -54}}, color = {0, 0, 127}));
  connect(speedSensorLoad.w, add.u2) annotation (
    Line(points = {{50, -51}, {50, -70}, {-176, -70}, {-176, -44.8}, {-153.6, -44.8}}, color = {0, 0, 127}));
  connect(firstOrder.y, speedController.u_s)
    annotation (Line(points={{-141.2,2},{-136,2}}, color={0,0,127}));
  connect(firstOrder.u, sine.y) annotation (Line(points={{-159.6,2},{-176,2},{
          -176,36},{-20,36},{-20,60},{-29,60}},                   color={0,0,
          127}));
  connect(speedController.y, add3.u2) annotation (Line(points={{-113,2},{-113,
          1.2},{-105,1.2}}, color={0,0,127}));
  connect(add3.y, limiter.u) annotation (Line(points={{-93.5,4.5},{-93.5,3},{
          -85.4,3}}, color={0,0,127}));
  connect(limiter.y, torqueMotor.tau) annotation (Line(points={{-69.3,3},{-64,3},
          {-64,0},{-57,0}}, color={0,0,127}));
  annotation (
    experiment(
      StopTime=4,
      Interval=0.001,
      __Dymola_Algorithm="Dassl"),
  Diagram(coordinateSystem(preserveAspectRatio = true, extent = {{-180, 100}, {120, -80}}), graphics = {Rectangle(lineColor = {255, 0, 0}, extent = {{-96, 76}, {-24, 44}}), Text(textColor = {255, 0, 0}, extent = {{-94, 87}, {-27, 79}}, textString = "reference speed generation"), Text(origin = {26, 0}, textColor = {255, 0, 0}, extent = {{-142, 24}, {-88, 18}}, textString = "controller"), Rectangle(origin = {137, 2}, lineColor = {255, 0, 0}, extent = {{-302, 16}, {-203, -16}})}),
    Documentation(info = "<html>
<p>
A&nbsp;controlled simple drive train to generate noisy signals of speed sensors.
The signals <code>noisyMotorSpeed</code> and <code>noisyLoadSpeed</code> can be
used as virtual measurements for
<a href=\"modelica://Credibility.Examples.SimpleControlledDriveNonlinear.SimpleControlledDrive_forCalibration\">model calibration</a>,
whereby this example is configured in a&nbsp;way especially suitable for calibration
of the damping parameter <code>damper.d</code>. Therefore, <code>damper.d</code>
is set to its nominal value of 100&nbsp;N&middot;m&middot;s/rad here, which is the
value a&nbsp;successful calibration shall striver.
</p>
<p>
A&nbsp;suitable reference motor speed for calibration of this drive train
configuration is prescribed by block <code>sine</code> which generates a&nbsp;sine
sweep signal of constant amplitude. It is used as a&nbsp;reference value of
the feedback controller which commands a&nbsp;motor torque for the drive train.
</p>
</html>"),
  Icon(coordinateSystem(extent = {{-200, -200}, {200, 200}}, preserveAspectRatio = true)));
end DC_Motor_Workshop;