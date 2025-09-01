within Credibility.Types;
type UncertaintyKind1D = enumeration(
    Interval1D,
    IntervalTolerance1D,
    Uniform1D,
    UniformTolerance1D,
    TruncatedNormal1D,
    TruncatedNormalTolerance1D) "Enumeration defining kind of uncertainty of 1D tables"
    annotation (
      Icon(graphics={
          Ellipse(
            extent={{-100,100},{100,-100}},
            lineColor={255,0,255}),
          Text(
            textColor={255,0,255},
            extent={{-90,-90},{90,90}},
            textString="e")}));
