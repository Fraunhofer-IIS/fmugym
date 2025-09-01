within Credibility.Types;
type SourceType = enumeration(
    Unknown
        "Unknown (source not known)",
    Estimated
        "Estimated (educated guess, from last project, extrapolated, ...)",
    Provided
        "Provided (data sheet, supplier specification, ...)",
    Computed
        "Computed (from CAD, CFD, FEM, from simulations, ...)",
    Measured
        "Measured (directly determined from verified measurements)",
    Calibrated
        "Calibrated (determined from verified measurements)")
    "Enumeration defining source of credibility information"
    annotation (
      Icon(graphics={
          Ellipse(
            extent={{-100,100},{100,-100}},
            lineColor={255,0,255}),
          Text(
            textColor={255,0,255},
            extent={{-90,-90},{90,90}},
            textString="e")}));
