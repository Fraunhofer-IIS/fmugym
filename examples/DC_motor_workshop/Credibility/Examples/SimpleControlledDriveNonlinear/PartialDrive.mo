within Credibility.Examples.SimpleControlledDriveNonlinear;
partial model PartialDrive "Not calibrated drive model with nonlinear compliance"

  replaceable parameter PartialData data
    "Data of the drive train, including credibility information"
    annotation (choicesAllMatching=true, Placement(transformation(extent={{60,30},{80,50}})));

  Modelica.Mechanics.Rotational.Components.Inertia inertiaMotor(
    J=data.Jm.value)
    "J from datasheet; Init defined by system spec or scenario spec"
    annotation (Placement(transformation(extent={{-32,-10},{-12,10}})));
  Modelica.Mechanics.Rotational.Sources.Torque torqueMotor annotation (Placement(transformation(extent={{-55,-10},{-35,10}})));
  Credibility.Examples.SimpleControlledDriveNonlinear.Utilities.SpringNonLinear spring(
    tableOnFile=false, table=data.stiffness.table)
    "table from datasheet; Init defined by system spec or scenario spec" annotation (Placement(transformation(extent={{26,22},{46,2}})));
  Modelica.Mechanics.Rotational.Components.Damper damper(
    d=data.damping.value,
    phi_rel(fixed=false),
    w_rel(fixed=true, start=0),
    a_rel(fixed=true, start=0),
    stateSelect=StateSelect.prefer)
    "d to be calibrated; Init defined by system spec or scenario spec" annotation (Placement(transformation(extent={{26,-20},{46,0}})));
  Modelica.Mechanics.Rotational.Components.Inertia inertiaLoad(
    J=data.Ja.value,
    phi(fixed=true, start=0),
    w(fixed=true, start=0))
    "Guess value, with defined uncertainty; Init defined by system spec or scenario spec"
    annotation (Placement(transformation(extent={{58,-10},{78,10}})));
  Modelica.Mechanics.Rotational.Sensors.SpeedSensor speedSensor annotation (
      Placement(transformation(extent={{10,-10},{-10,10}},
        rotation=90,
        origin={-9,-22})));
  Modelica.Mechanics.Rotational.Components.IdealGear idealGear(ratio=data.ratio.value)
    annotation (Placement(transformation(extent={{-4,-10},{16,10}})));

equation
  connect(damper.flange_b, inertiaLoad.flange_a) annotation (Line(points={{46,-10},{50,-10},{50,0},{58,0}}));
  connect(idealGear.flange_b, damper.flange_a) annotation (Line(points={{16,0},{20,0},{20,-10},{26,-10}},
        color={0,0,0}));
  connect(spring.flange_b, inertiaLoad.flange_a) annotation (Line(points={{46,12},{50,12},{50,0},{58,0}},
        color={0,0,0}));
  connect(torqueMotor.flange,inertiaMotor. flange_a) annotation (Line(points={{-35,0},{-32,0}}));
  connect(speedSensor.flange,inertiaMotor. flange_b) annotation (Line(points={{-9,-12},{-9,0},{-12,0}}));
  connect(inertiaMotor.flange_b, idealGear.flange_a) annotation (Line(points={{-12,0},{-4,0}},
        color={0,0,0}));
  connect(spring.flange_a, damper.flange_a) annotation (Line(points={{26,12},{20,12},{20,-10},{26,-10}},
        color={0,0,0}));

  annotation (
    Icon(coordinateSystem(extent={{-100,-100},{110,100}})),
    Diagram(
      coordinateSystem(
        preserveAspectRatio=true,
        extent={{-100,-100},{100,100}}),
      graphics={
        Rectangle(
          extent={{-60,20},{80,-34}},
          lineColor={255,0,0}),
        Text(
          extent={{-29,28},{36,22}},
          textColor={255,0,0},
          textString="plant (simple drive train)")}),
    Documentation(info="<html>
<p>
This is a&nbsp;simple drive train consisting of the following elements:
</p>

<ul>
  <li>
    A&nbsp;motor inertia that is driven by an external torque.
  </li>
  <li>
    The measured motor speed (=&nbsp;angular velocity) provided as output signal.
  </li>
  <li>
    A&nbsp;load inertia.
  </li>
  <li>
    A&nbsp;gearbox. The compliance of the
    gearbox is modelled with <strong>a&nbsp;nonlinear spring defined with
    a&nbsp;table and a&nbsp;linear damper</strong>. The compliance is modelled
    because it limits the performance of a&nbsp;controller.
  </li>
</ul>

<p>
Typically, a&nbsp;controller is designed to compute the motor torque, from
the measured motor speed and a&nbsp;provided reference speed of the motor
(= reference speed of the load, because the gear ratio is one).
Furthermore, an external torque might be applied on the load inertia.
</p>

<p>
All the significant parameters of the drive train are provided via the 
record instance <code>data</code> containing the values of the
parameters, as well as information about the credibility of the
parameters.
</p>


<h4>Initialization</h4>
<p>
The initialization is a&nbsp;bit tricky. The goal is to initialize in steady state
because initial transients are then no longer present.
However, this depends on the controller that is generating the motor torque.
Furthermore, when exporting the model as FMU, the FMU must be also reasonably 
initialized. Since this partial model has four states (e.g., <code>inertiaLoad.phi</code>,
<code>inertiaLoad.w</code>, <code>damper.phi_rel</code> and <code>damper.w_rel</code>),
four initial conditions have to be provided.
Typically, the user is interested in the load position/speed,
not in the motor position/speed. Therefore, the following initialization is defined:
</p>

<ul>
  <li>
    The load angle and load angular velocity are both initialized to zero.
    Note, if standard steady state initialization would be performed,
    then the load angle would be undefined.
    When a&nbsp;controller is added to the model and steady-state initialization
    is desired, then the load angular acceleration should also be set to zero,
    provided the controller has at least one state (e.g. a&nbsp;PI controller)
    that can be computed from this setting, cf. initialization of
    <a href=\"modelica://Credibility.Examples.SimpleControlledDriveNonlinear.SimpleControlledDrive_original\">SimpleControlledDrive_original</a>.
  </li>
  <li>
    The relative angular velocity and relative angular acceleration of the
    gear/compliance are both initialized to zero, see <code>damper</code>
    element which is, therefore, initialized in steady state.
  </li>
</ul>
</html>"));
end PartialDrive;
