within Credibility.Types;
type UncertaintyKind = enumeration(
    Interval,
    IntervalTolerance,
    Uniform,
    UniformTolerance,
    TruncatedNormal,
    TruncatedNormalTolerance) "Enumeration defining kind of uncertainty of scalar real parameter"
    annotation (
      Icon(graphics={
          Ellipse(
            extent={{-100,100},{100,-100}},
            lineColor={255,0,255}),
          Text(
            textColor={255,0,255},
            extent={{-90,-90},{90,90}},
            textString="e")}));
