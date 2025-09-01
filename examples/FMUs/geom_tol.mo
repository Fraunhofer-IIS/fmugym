model geom_tol
  Modelica.Blocks.Interfaces.RealOutput FR1_min annotation(
    Placement(transformation(origin = {84, 114}, extent = {{-10, -10}, {10, 10}}), iconTransformation(extent = {{-10, -10}, {10, 10}})));
  Modelica.Blocks.Interfaces.RealOutput FR1_max annotation(
    Placement(transformation(origin = {84, 76}, extent = {{-10, -10}, {10, 10}}), iconTransformation(extent = {{-10, -10}, {10, 10}})));
  Modelica.Blocks.Interfaces.RealInput R1(start=0.1) annotation(
    Placement(transformation(origin = {-140, 146}, extent = {{-20, -20}, {20, 20}}), iconTransformation(origin = {-62, 90}, extent = {{-20, -20}, {20, 20}})));
  Modelica.Blocks.Interfaces.RealInput S2(start=0.1) annotation(
    Placement(transformation(origin = {-140, 111}, extent = {{-20, -20}, {20, 20}}), iconTransformation(origin = {-54, -5}, extent = {{-20, -20}, {20, 20}})));
  Modelica.Blocks.Interfaces.RealInput C4(start=0.1) annotation(
    Placement(transformation(origin = {-140, 69}, extent = {{-20, -20}, {20, 20}}), iconTransformation(origin = {-24, -169}, extent = {{-20, -20}, {20, 20}})));
  Modelica.Blocks.Interfaces.RealInput S4(start=0.2) annotation(
    Placement(transformation(origin = {-140, 22}, extent = {{-20, -20}, {20, 20}}), iconTransformation(origin = {68, -270}, extent = {{-20, -20}, {20, 20}})));
  Modelica.Blocks.Interfaces.RealInput C5(start=0.2) annotation(
    Placement(transformation(origin = {-138, -25}, extent = {{-20, -20}, {20, 20}}), iconTransformation(origin = {168, -333}, extent = {{-20, -20}, {20, 20}})));
  Modelica.Blocks.Interfaces.RealInput R4(start=0.1) annotation(
    Placement(transformation(origin = {-137, -68}, extent = {{-20, -20}, {20, 20}}), iconTransformation(origin = {239, -410}, extent = {{-20, -20}, {20, 20}})));
  Modelica.Blocks.Interfaces.RealInput N2(start=0.1) annotation(
    Placement(transformation(origin = {-137, -112}, extent = {{-20, -20}, {20, 20}}), iconTransformation(origin = {323, -506}, extent = {{-20, -20}, {20, 20}})));
  Modelica.Blocks.Interfaces.RealOutput FR2_min annotation(
    Placement(transformation(origin = {84, -29}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {18, -57}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Blocks.Interfaces.RealOutput FR2_max annotation(
    Placement(transformation(origin = {84, -65}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {20, -81}, extent = {{-10, -10}, {10, 10}})));

protected 
  parameter Real Char_L_S4 = 27.5;
  parameter Real Char_L_R4 = 17.2679;
  parameter Real l_R = 30.5;
  parameter Real r_B = 1.0;
  parameter Real a_C = 10.0;

  parameter Real Pos_FR1_R_x = 13.96;
  parameter Real Pos_FR1_R_y = 7.25;
  parameter Real Pos_FR2_RN_x = 31.25;
  parameter Real Pos_FR2_RN_y = 1.25;
  parameter Real Pos_FR2_N_x = 13.21;
  parameter Real Pos_FR2_N_y = 1.25;
  parameter Real Pos_P_R_x = 31.5;
  parameter Real Pos_P_R_y = 0.0;
  parameter Real Pos_FR2ref_C_x = -0.25;
  parameter Real Pos_FR2ref_C_y = 1.25;

  parameter Real FR1_nom = 0.5;
  parameter Real FR2_nom = 0.15;

  Real alpha_C4;
  Real S4_pos;
  Boolean S4_negative;
  Real beta_S4;
  Real alpha_S4;
  Real FR1_S4;
  Real XM, YM, XB, YB;
  Real FR2_S4;
  Real gamma_R4;
  Real FR2_R4;

equation
  // C4 Effect
  alpha_C4 = asin(C4/2 / (l_R + r_B));

  // S4 sign logic
  S4_negative = S4 < 0;
  S4_pos = if S4_negative then -S4 else S4;

  beta_S4 = atan(S4_pos / Char_L_S4);
  alpha_S4 = acos((a_C * cos(beta_S4) + l_R + r_B - a_C) / (l_R + r_B));

  FR1_S4 = (Pos_FR1_R_y*(1 - cos(alpha_S4)) + Pos_FR1_R_x*sin(alpha_S4)) * (if S4_negative then -1 else 1);

  XM = Pos_FR2_RN_x * cos(alpha_S4) - Pos_FR2_RN_y * sin(alpha_S4);
  YM = Pos_FR2_RN_x * sin(alpha_S4) + Pos_FR2_RN_y * cos(alpha_S4);
  XB = Pos_P_R_x * cos(alpha_S4) - Pos_P_R_y * sin(alpha_S4) + Pos_FR2ref_C_x * cos(beta_S4) - Pos_FR2ref_C_y * sin(beta_S4);
  YB = Pos_P_R_x * sin(alpha_S4) + Pos_P_R_y * cos(alpha_S4) + Pos_FR2ref_C_x * sin(beta_S4) + Pos_FR2ref_C_y * cos(beta_S4);
  FR2_S4 = sqrt((XB - XM)^2 + (YB - YM)^2) * (if S4_negative then -1 else 1);

  gamma_R4 = atan(R4 / Char_L_R4);
  FR2_R4 = Pos_FR2_N_y*(1 - cos(gamma_R4)) + Pos_FR2_N_x*sin(gamma_R4);

  // Final outputs
  FR1_min = FR1_nom - R1/4 - S2/4
                - (Pos_FR1_R_y*(1 - cos(alpha_C4)) + Pos_FR1_R_x*sin(alpha_C4))
                - FR1_S4;

  FR1_max = FR1_nom + R1/4 + S2/4
                + (Pos_FR1_R_y*(1 - cos(alpha_C4)) + Pos_FR1_R_x*sin(alpha_C4))
                + FR1_S4;

  FR2_min = FR2_nom - C5/4 - N2/4
                - (Pos_FR2_RN_y*(1 - cos(alpha_C4)) + Pos_FR2_RN_x*sin(alpha_C4))
                - FR2_S4
                - FR2_R4;

  FR2_max = FR2_nom + C5/4 + N2/4
                + (Pos_FR2_RN_y*(1 - cos(alpha_C4)) + Pos_FR2_RN_x*sin(alpha_C4))
                + FR2_S4
                + FR2_R4;

annotation(Inline=true);

end geom_tol;